"""Minimal reproduction of the y-axis inversion bug in Genesis DLS IK.

Symptom: commanding dy=-0.45 moves the Franka EE in the +y direction.

This script tests:
  1. DLS IK (manual Jacobian + damped least squares) — the method used in ScoopTransferTask
  2. Genesis built-in robot.inverse_kinematics() (if available)
  3. Two target definitions: link origin vs scoop_tip offset

For each combination, logs commanded vs actual delta-y and whether signs match.

Usage:
    source /home/zhuzihou/dev/Genesis/.venv/bin/activate
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
    python pta/scripts/ik_minimal_repro.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch

import genesis as gs


# ── Config ──────────────────────────────────────────────────────────────

HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
DY_VALUES = [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3]
DLS_LAMBDA = 0.01
IK_ITERS = 50  # iterations for DLS convergence


# ── Scene setup ─────────────────────────────────────────────────────────

def build_scene():
    """Create minimal Genesis scene: 1 Franka (scoop), ground, no particles."""
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=10),
        rigid_options=gs.options.RigidOptions(
            dt=2e-3,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=False,
    )

    # Ground
    scene.add_entity(
        material=gs.materials.Rigid(needs_coup=False),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    # Franka with scoop
    robot = scene.add_entity(
        material=gs.materials.Rigid(needs_coup=False),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_scoop.xml",
            pos=(0.0, 0.0, 0.0),
        ),
    )

    scene.build(n_envs=0)

    # PD gains (same as SceneBuilder)
    robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]))
    robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200]))
    robot.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12]),
        np.array([87, 87, 87, 87, 12, 12, 12]),
    )

    return scene, robot


def get_ee_pos(link):
    """Return EE position as 1D tensor."""
    p = link.get_pos()
    if p.dim() > 1:
        p = p.squeeze(0)
    return p


# ── DLS IK ──────────────────────────────────────────────────────────────

def dls_ik_step(robot, ee_link, target_pos, current_qpos, lam=DLS_LAMBDA):
    """One step of damped-least-squares IK (position only, 3-DOF target).

    Returns new qpos (7,) tensor.
    """
    ee_pos = get_ee_pos(ee_link)
    delta_x = target_pos - ee_pos  # (3,)

    # We need 6-DOF delta for the Jacobian (pad rotation with zeros)
    delta_pose = torch.cat([delta_x, torch.zeros(3, device=gs.device)])  # (6,)

    jacobian = robot.get_jacobian(link=ee_link)  # (6, n_dof)
    if jacobian.dim() == 3:
        jacobian = jacobian.squeeze(0)

    jac_T = jacobian.T  # (n_dof, 6)
    lam_I = (lam ** 2) * torch.eye(jacobian.shape[0], device=gs.device)
    delta_q = jac_T @ torch.inverse(jacobian @ jac_T + lam_I) @ delta_pose  # (n_dof,)

    new_qpos = current_qpos + delta_q
    return new_qpos


def dls_ik_solve(robot, scene, ee_link, target_pos, start_qpos, n_iters=IK_ITERS):
    """Iteratively solve DLS IK, applying qpos and stepping each iteration."""
    qpos = start_qpos.clone()
    for _ in range(n_iters):
        qpos = dls_ik_step(robot, ee_link, target_pos, qpos)
        robot.set_qpos(qpos)
        scene.step()
    return qpos, get_ee_pos(ee_link)


# ── Single-step DLS (as used in the task) ───────────────────────────────

def dls_ik_single_step(robot, scene, ee_link, delta_pos, current_qpos):
    """Exactly replicates ScoopTransferTask._compute_ik for position-only delta.

    This is the method actually called during RL: ONE step of DLS, then
    control_dofs_position + scene.step.
    """
    ee_pos = get_ee_pos(ee_link)
    target_pos = ee_pos + delta_pos

    # Build 6-DOF delta (rotation = 0)
    delta_rot = torch.zeros(3, device=gs.device)
    delta_pose = torch.cat([delta_pos, delta_rot])  # (6,)

    jacobian = robot.get_jacobian(link=ee_link)
    if jacobian.dim() == 3:
        jacobian = jacobian.squeeze(0)

    jac_T = jacobian.T
    lam = DLS_LAMBDA
    lam_I = (lam ** 2) * torch.eye(jacobian.shape[0], device=gs.device)
    delta_q = jac_T @ torch.inverse(jacobian @ jac_T + lam_I) @ delta_pose

    new_qpos = current_qpos + delta_q

    # Apply via PD controller (same as task.step)
    robot.control_dofs_position(new_qpos)
    for _ in range(10):  # settle
        scene.step()

    return new_qpos, get_ee_pos(ee_link), target_pos


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("IK Minimal Repro: Y-Axis Inversion Bug")
    print("=" * 80)

    scene, robot = build_scene()
    ee_link = robot.get_link("scoop")

    # Also get link7 and hand-equivalent for comparison
    link7 = robot.get_link("link7")

    # Set home pose
    home_qpos = torch.tensor(HOME_QPOS, dtype=torch.float32, device=gs.device)
    robot.set_qpos(home_qpos)
    for _ in range(20):
        scene.step()

    start_pos = get_ee_pos(ee_link).clone()
    start_pos_link7 = get_ee_pos(link7).clone()
    print(f"\nHome qpos: {HOME_QPOS}")
    print(f"EE (scoop link) start pos: ({start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.4f})")
    print(f"Link7 start pos:           ({start_pos_link7[0]:.4f}, {start_pos_link7[1]:.4f}, {start_pos_link7[2]:.4f})")

    # ── Test 1: DLS IK single-step (as used in task) on scoop link ──────
    print("\n" + "=" * 80)
    print("TEST 1: DLS IK single-step (task._compute_ik equivalent) — scoop link")
    print("=" * 80)
    print(f"{'dy_cmd':>10s} {'target_y':>10s} {'actual_y':>10s} {'dy_actual':>10s} {'sign_match':>10s} {'ratio':>10s}")
    print("-" * 70)

    results_1 = []
    for dy in DY_VALUES:
        # Reset to home
        robot.set_qpos(home_qpos)
        for _ in range(10):
            scene.step()

        delta_pos = torch.tensor([0.0, dy, 0.0], dtype=torch.float32, device=gs.device)
        new_qpos, actual_pos, target_pos = dls_ik_single_step(
            robot, scene, ee_link, delta_pos, home_qpos
        )

        dy_actual = (actual_pos[1] - start_pos[1]).item()
        sign_match = (dy > 0 and dy_actual > 0) or (dy < 0 and dy_actual < 0) or (dy == 0)
        ratio = dy_actual / dy if abs(dy) > 1e-6 else float('nan')

        print(f"{dy:>10.3f} {target_pos[1].item():>10.4f} {actual_pos[1].item():>10.4f} {dy_actual:>10.4f} {'YES' if sign_match else '** NO **':>10s} {ratio:>10.3f}")
        results_1.append({
            "dy_cmd": dy, "target_y": target_pos[1].item(),
            "actual_y": actual_pos[1].item(), "dy_actual": dy_actual,
            "sign_match": sign_match, "ratio": ratio,
        })

    # ── Test 2: DLS IK iterative solve on scoop link ────────────────────
    print("\n" + "=" * 80)
    print("TEST 2: DLS IK iterative (50 iters) — scoop link")
    print("=" * 80)
    print(f"{'dy_cmd':>10s} {'target_y':>10s} {'actual_y':>10s} {'dy_actual':>10s} {'sign_match':>10s} {'error':>10s}")
    print("-" * 70)

    results_2 = []
    for dy in DY_VALUES:
        robot.set_qpos(home_qpos)
        for _ in range(10):
            scene.step()

        target_pos = start_pos.clone()
        target_pos[1] += dy

        _, actual_pos = dls_ik_solve(robot, scene, ee_link, target_pos, home_qpos, n_iters=50)

        dy_actual = (actual_pos[1] - start_pos[1]).item()
        sign_match = (dy > 0 and dy_actual > 0) or (dy < 0 and dy_actual < 0) or (dy == 0)
        error = abs(actual_pos[1].item() - target_pos[1].item())

        print(f"{dy:>10.3f} {target_pos[1].item():>10.4f} {actual_pos[1].item():>10.4f} {dy_actual:>10.4f} {'YES' if sign_match else '** NO **':>10s} {error:>10.4f}")
        results_2.append({
            "dy_cmd": dy, "target_y": target_pos[1].item(),
            "actual_y": actual_pos[1].item(), "dy_actual": dy_actual,
            "sign_match": sign_match, "error": error,
        })

    # ── Test 3: DLS IK single-step on link7 (no scoop offset) ──────────
    print("\n" + "=" * 80)
    print("TEST 3: DLS IK single-step — link7 (no scoop body)")
    print("=" * 80)
    print(f"{'dy_cmd':>10s} {'target_y':>10s} {'actual_y':>10s} {'dy_actual':>10s} {'sign_match':>10s} {'ratio':>10s}")
    print("-" * 70)

    results_3 = []
    for dy in DY_VALUES:
        robot.set_qpos(home_qpos)
        for _ in range(10):
            scene.step()

        start_l7 = get_ee_pos(link7).clone()
        delta_pos = torch.tensor([0.0, dy, 0.0], dtype=torch.float32, device=gs.device)

        # DLS IK targeting link7
        ee_pos = get_ee_pos(link7)
        target_pos_l7 = ee_pos + delta_pos
        delta_rot = torch.zeros(3, device=gs.device)
        delta_pose = torch.cat([delta_pos, delta_rot])

        jacobian = robot.get_jacobian(link=link7)
        if jacobian.dim() == 3:
            jacobian = jacobian.squeeze(0)
        jac_T = jacobian.T
        lam_I = (DLS_LAMBDA ** 2) * torch.eye(jacobian.shape[0], device=gs.device)
        delta_q = jac_T @ torch.inverse(jacobian @ jac_T + lam_I) @ delta_pose
        new_qpos = home_qpos + delta_q

        robot.control_dofs_position(new_qpos)
        for _ in range(10):
            scene.step()

        actual_pos = get_ee_pos(link7)
        dy_actual = (actual_pos[1] - start_l7[1]).item()
        sign_match = (dy > 0 and dy_actual > 0) or (dy < 0 and dy_actual < 0) or (dy == 0)
        ratio = dy_actual / dy if abs(dy) > 1e-6 else float('nan')

        print(f"{dy:>10.3f} {target_pos_l7[1].item():>10.4f} {actual_pos[1].item():>10.4f} {dy_actual:>10.4f} {'YES' if sign_match else '** NO **':>10s} {ratio:>10.3f}")
        results_3.append({
            "dy_cmd": dy, "target_y": target_pos_l7[1].item(),
            "actual_y": actual_pos[1].item(), "dy_actual": dy_actual,
            "sign_match": sign_match, "ratio": ratio,
        })

    # ── Test 4: Genesis built-in IK on scoop link ──────────────────────
    print("\n" + "=" * 80)
    print("TEST 4: Genesis built-in robot.inverse_kinematics() — scoop link")
    print("=" * 80)

    results_4 = []
    try:
        # Test if built-in IK is available
        robot.set_qpos(home_qpos)
        for _ in range(10):
            scene.step()

        arm_dof_idx = torch.arange(robot.n_dofs, device=gs.device)
        ee_quat = ee_link.get_quat()
        if ee_quat.dim() > 1:
            ee_quat = ee_quat.squeeze(0)

        print(f"{'dy_cmd':>10s} {'target_y':>10s} {'actual_y':>10s} {'dy_actual':>10s} {'sign_match':>10s} {'error':>10s}")
        print("-" * 70)

        for dy in DY_VALUES:
            robot.set_qpos(home_qpos)
            for _ in range(10):
                scene.step()

            target_pos = start_pos.clone()
            target_pos[1] += dy

            qpos = robot.inverse_kinematics(
                link=ee_link,
                pos=target_pos,
                quat=ee_quat,
                dofs_idx_local=arm_dof_idx,
            )

            robot.set_qpos(qpos)
            for _ in range(10):
                scene.step()

            actual_pos = get_ee_pos(ee_link)
            dy_actual = (actual_pos[1] - start_pos[1]).item()
            sign_match = (dy > 0 and dy_actual > 0) or (dy < 0 and dy_actual < 0) or (dy == 0)
            error = abs(actual_pos[1].item() - target_pos[1].item())

            print(f"{dy:>10.3f} {target_pos[1].item():>10.4f} {actual_pos[1].item():>10.4f} {dy_actual:>10.4f} {'YES' if sign_match else '** NO **':>10s} {error:>10.4f}")
            results_4.append({
                "dy_cmd": dy, "target_y": target_pos[1].item(),
                "actual_y": actual_pos[1].item(), "dy_actual": dy_actual,
                "sign_match": sign_match, "error": error,
            })

    except Exception as e:
        print(f"  Built-in IK not available or failed: {e}")
        traceback.print_exc()

    # ── Test 5: Jacobian inspection ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("TEST 5: Jacobian inspection at home pose")
    print("=" * 80)

    robot.set_qpos(home_qpos)
    for _ in range(10):
        scene.step()

    jac_scoop = robot.get_jacobian(link=ee_link)
    if jac_scoop.dim() == 3:
        jac_scoop = jac_scoop.squeeze(0)

    jac_link7 = robot.get_jacobian(link=link7)
    if jac_link7.dim() == 3:
        jac_link7 = jac_link7.squeeze(0)

    print("\nJacobian (scoop link) — row 1 (y-translation):")
    print(f"  {jac_scoop[1].cpu().numpy()}")

    print("\nJacobian (link7) — row 1 (y-translation):")
    print(f"  {jac_link7[1].cpu().numpy()}")

    print("\nJacobian (scoop link) — full linear part (rows 0-2):")
    for i, label in enumerate(["x", "y", "z"]):
        print(f"  {label}: {jac_scoop[i].cpu().numpy()}")

    print("\nJacobian (link7) — full linear part (rows 0-2):")
    for i, label in enumerate(["x", "y", "z"]):
        print(f"  {label}: {jac_link7[i].cpu().numpy()}")

    # Check sign of J[1,0] — this tells us if +dq1 -> +dy or -dy
    print(f"\n  J_scoop[y, joint1] = {jac_scoop[1, 0].item():.6f}  (should be positive if +dq1 -> +dy)")
    print(f"  J_link7[y, joint1] = {jac_link7[1, 0].item():.6f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    def summarize(name, results):
        n_match = sum(1 for r in results if r["sign_match"])
        n_total = len(results)
        neg_match = sum(1 for r in results if r["dy_cmd"] < 0 and r["sign_match"])
        neg_total = sum(1 for r in results if r["dy_cmd"] < 0)
        pos_match = sum(1 for r in results if r["dy_cmd"] > 0 and r["sign_match"])
        pos_total = sum(1 for r in results if r["dy_cmd"] > 0)
        print(f"  {name}: {n_match}/{n_total} sign matches (neg: {neg_match}/{neg_total}, pos: {pos_match}/{pos_total})")

    summarize("Test 1 — DLS single-step, scoop link", results_1)
    summarize("Test 2 — DLS iterative, scoop link", results_2)
    summarize("Test 3 — DLS single-step, link7", results_3)
    if results_4:
        summarize("Test 4 — Genesis built-in IK, scoop link", results_4)

    # Diagnosis
    t1_inversions = [r for r in results_1 if not r["sign_match"]]
    t3_inversions = [r for r in results_3 if not r["sign_match"]]

    print("\n  DIAGNOSIS:")
    if t1_inversions and not t3_inversions:
        print("  -> Y-inversion is SCOOP-LINK SPECIFIC (link7 is fine)")
        print("  -> Root cause: the scoop body frame has a rotation (quat=0.383 0 0 0.924)")
        print("     that rotates the Jacobian reference frame, mixing/flipping axes.")
        print("  -> Fix: use link7 as the IK target link, or apply a TCP offset.")
    elif t1_inversions and t3_inversions:
        print("  -> Y-inversion affects BOTH scoop and link7 -> likely a Genesis IK bug")
    elif not t1_inversions and not t3_inversions:
        print("  -> NO y-inversion detected in single-step tests")
        print("     (Bug may only manifest with larger deltas or accumulated drift)")
    else:
        print("  -> Unexpected pattern — see detailed results above")


if __name__ == "__main__":
    main()
