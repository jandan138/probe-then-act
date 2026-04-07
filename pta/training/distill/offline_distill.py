"""Offline distillation -- student learns from pre-collected teacher data.

Pipeline:
  1. collect_teacher_demos: roll out a trained teacher (M8) and save
     observation/action pairs to disk as .npz.
  2. bc_pretrain: behavioural cloning from demos to warm-start the student
     policy (future work -- documented stub).
  3. load_bc_pretrained_into_ppo: transfer BC-pretrained weights into an
     SB3 PPO model so fine-tuning can continue with RL.

For the initial M7 results, only step 1 is needed.  The student can train
end-to-end with PPO (see train_m7.py).  BC pre-training is a planned
enhancement to accelerate convergence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Step 1: Collect teacher demonstrations
# ---------------------------------------------------------------------------

def collect_teacher_demos(
    model_path: str,
    env_fn,
    n_episodes: int = 50,
    save_path: Optional[str] = None,
    deterministic: bool = True,
) -> Dict[str, np.ndarray]:
    """Roll out a trained SB3 teacher and collect obs/action pairs.

    Parameters
    ----------
    model_path : str
        Path to the SB3 model checkpoint (.zip).
    env_fn : callable
        Zero-argument callable that returns a Gymnasium env compatible
        with the teacher model (should include PrivilegedObsWrapper for
        M8 demos, or the standard stack for M1).
    n_episodes : int
        Number of episodes to collect.
    save_path : str, optional
        If provided, save the demo dataset as a .npz file.
    deterministic : bool
        Whether to use deterministic actions (True = greedy, no noise).

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``observations`` (N, obs_dim), ``actions`` (N, act_dim),
        ``rewards`` (N,), ``episode_starts`` (N,) -- boolean mask for
        the first step of each episode.
    """
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = env_fn()

    all_obs = []
    all_actions = []
    all_rewards = []
    all_episode_starts = []

    for ep in range(n_episodes):
        obs, _info = env.reset()
        done = False
        step_in_ep = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            all_obs.append(obs.copy())
            all_actions.append(action.copy())
            all_episode_starts.append(step_in_ep == 0)

            obs, reward, terminated, truncated, _info = env.step(action)
            all_rewards.append(reward)
            done = terminated or truncated
            step_in_ep += 1

        if (ep + 1) % 10 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes "
                  f"({sum(len(o) for o in [all_obs])} transitions)")

    env.close()

    dataset = {
        "observations": np.array(all_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "episode_starts": np.array(all_episode_starts, dtype=bool),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(save_path), **dataset)
        print(f"Saved {len(all_obs)} transitions to {save_path}")

    return dataset


# ---------------------------------------------------------------------------
# Step 2: Behavioural cloning pre-training (stub -- future work)
# ---------------------------------------------------------------------------

def bc_pretrain(
    demos: Dict[str, np.ndarray],
    policy_net: torch.nn.Module,
    n_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "auto",
) -> Dict[str, Any]:
    """Pre-train a policy network via behavioural cloning on teacher demos.

    This is a planned enhancement.  For the initial M7 results, the
    student trains end-to-end with PPO from scratch.

    Parameters
    ----------
    demos : dict[str, np.ndarray]
        Dataset from ``collect_teacher_demos`` (or loaded .npz).
    policy_net : torch.nn.Module
        The policy network to train (e.g., SB3 policy's ``mlp_extractor``
        + ``action_net``).
    n_epochs : int
        Number of supervised training epochs.
    lr : float
        Learning rate for Adam.
    batch_size : int
        Mini-batch size.
    device : str
        Torch device.

    Returns
    -------
    dict[str, Any]
        Training summary: ``{"final_loss": float, "losses": list[float]}``.

    Raises
    ------
    NotImplementedError
        This is a stub for future work.
    """
    # TODO(M7-v2): Implement BC pre-training.
    #
    # Planned approach:
    #   1. Build a DataLoader from demos["observations"] and demos["actions"]
    #   2. MSE loss between policy_net(obs) and demo actions
    #   3. Train for n_epochs with Adam
    #   4. Return loss curve
    #
    # Note: if the teacher (M8) has privileged obs and the student (M7)
    # does not, we need to either:
    #   (a) strip privileged dims from demo observations, or
    #   (b) collect demos in the student obs space (no PrivilegedObsWrapper)
    #       but with teacher actions mapped through the student's wrapper
    #       stack.  Option (b) is cleaner.
    raise NotImplementedError(
        "BC pre-training is planned for M7-v2. "
        "For initial results, use end-to-end PPO (train_m7.py)."
    )


# ---------------------------------------------------------------------------
# Step 3: Transfer BC weights into SB3 PPO (stub -- future work)
# ---------------------------------------------------------------------------

def load_bc_pretrained_into_ppo(
    bc_state_dict: Dict[str, torch.Tensor],
    ppo_model,
) -> None:
    """Transfer BC-pretrained weights into an SB3 PPO model.

    This loads the policy network weights from BC pre-training into the
    corresponding layers of an SB3 PPO model, allowing RL fine-tuning
    from a warm start.

    Parameters
    ----------
    bc_state_dict : dict[str, torch.Tensor]
        State dict from the BC-pretrained policy network.
    ppo_model : stable_baselines3.PPO
        The PPO model whose policy weights will be updated.

    Raises
    ------
    NotImplementedError
        This is a stub for future work.
    """
    # TODO(M7-v2): Implement weight transfer.
    #
    # Planned approach:
    #   1. Map BC state_dict keys to SB3 policy keys
    #      (SB3 uses policy.mlp_extractor.policy_net.{0,2}.{weight,bias}
    #       and policy.action_net.{weight,bias})
    #   2. ppo_model.policy.load_state_dict(mapped_dict, strict=False)
    #   3. Optionally freeze value function (train only policy head)
    raise NotImplementedError(
        "BC-to-PPO weight transfer is planned for M7-v2. "
        "For initial results, use end-to-end PPO (train_m7.py)."
    )


# ---------------------------------------------------------------------------
# Legacy entry point (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def offline_distillation(
    teacher_data: Any,
    student: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform offline distillation from pre-collected teacher rollouts.

    Deprecated -- use ``collect_teacher_demos`` + ``bc_pretrain`` +
    ``load_bc_pretrained_into_ppo`` instead.

    Parameters
    ----------
    teacher_data : Any
        Pre-collected dataset of teacher rollouts (observations, actions,
        action distributions).
    student : Any
        Student policy to be trained.
    config : dict, optional
        Distillation hyper-parameters (lr, epochs, batch_size, ...).

    Returns
    -------
    dict[str, Any]
        Training summary including loss curves and final metrics.
    """
    raise NotImplementedError(
        "Use the new modular API: collect_teacher_demos, bc_pretrain, "
        "load_bc_pretrained_into_ppo. See train_m7.py for the recommended "
        "end-to-end PPO workflow."
    )
