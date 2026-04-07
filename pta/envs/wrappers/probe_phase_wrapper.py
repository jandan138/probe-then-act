"""Probe-phase wrapper for M7 (Probe-Then-Act).

Wraps a JointResidualWrapper environment.  At the start of each episode,
executes a fixed number of "probe" steps using the base trajectory, collects
observation traces, encodes them through the LatentBeliefEncoder to produce
a latent belief vector z, and appends z to every observation for the
remainder of the episode.

Unlike FixedProbeWrapper (which uses Cartesian probe actions), this wrapper
operates in joint space and reuses the first N steps of the base trajectory
as the probe phase.  The key insight is that the initial approach/contact
segment of the scripted trajectory already provides material-informative
force/position responses.

Ablation modes:
  - ``"none"``     : full M7 (probe + belief encoding)
  - ``"no_probe"`` : skip probe, z = zeros (tests if probing matters)
  - ``"no_belief"``: run probe but z = zeros (tests if belief matters)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
import torch


class ProbePhaseWrapper(gymnasium.Wrapper):
    """Execute probe steps at reset, encode traces into z, append to obs.

    Parameters
    ----------
    env:
        A JointResidualWrapper (or compatible).
    latent_dim:
        Dimensionality of the belief vector z appended to observations.
    n_probes:
        Number of probe steps executed at the start of each episode.
        During probing, the policy action is zeros (pure base trajectory).
    belief_encoder:
        Optional pre-trained LatentBeliefEncoder.  If None, a new one is
        created with default architecture.
    ablation:
        Ablation mode: ``"none"`` (full), ``"no_probe"``, ``"no_belief"``.
    trace_dim:
        Observation dimensionality of the inner env (auto-detected if None).
    device:
        Torch device for the belief encoder.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        latent_dim: int = 16,
        n_probes: int = 3,
        belief_encoder: Optional[Any] = None,
        ablation: str = "none",
        trace_dim: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env)

        assert ablation in ("none", "no_probe", "no_belief"), (
            f"Unknown ablation mode: {ablation!r}"
        )

        self.latent_dim = latent_dim
        self.n_probes = n_probes
        self.ablation = ablation
        self._device = device

        # Detect inner obs dim
        self._inner_obs_dim = trace_dim or env.observation_space.shape[0]

        # Build or use provided belief encoder
        if belief_encoder is not None:
            self._belief_encoder = belief_encoder
        else:
            from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder
            self._belief_encoder = LatentBeliefEncoder(
                trace_dim=self._inner_obs_dim,
                latent_dim=latent_dim,
                hidden_dim=128,
                num_layers=2,
            )
        self._belief_encoder = self._belief_encoder.to(self._device)
        # Belief encoder is frozen during PPO; updated separately or end-to-end
        self._belief_encoder.eval()

        # Expand observation space: inner_obs + z
        total_dim = self._inner_obs_dim + self.latent_dim
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

        # Cache
        self._z: Optional[np.ndarray] = None

    def _encode_traces(self, traces: np.ndarray) -> np.ndarray:
        """Encode probe traces into a latent belief vector z.

        Parameters
        ----------
        traces : np.ndarray
            Shape ``(N, obs_dim)`` — observations from probe steps.

        Returns
        -------
        np.ndarray
            Belief vector z of shape ``(latent_dim,)``.
        """
        # traces: (N, obs_dim) -> (1, N, obs_dim) batch
        t = torch.tensor(traces, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            z, _sigma = self._belief_encoder(t)
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Append cached z to observation."""
        z = self._z if self._z is not None else np.zeros(self.latent_dim, dtype=np.float32)
        return np.concatenate([obs, z]).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        if self.ablation == "no_probe":
            # Skip probing entirely -- z = zeros
            self._z = np.zeros(self.latent_dim, dtype=np.float32)
            info["probe_steps"] = 0
            return self._augment_obs(obs), info

        # Execute probe phase: step with zero residuals (pure base trajectory)
        probe_traces = [obs.copy()]
        zero_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        total_probe_reward = 0.0

        for _ in range(self.n_probes):
            obs, reward, terminated, truncated, step_info = self.env.step(zero_action)
            probe_traces.append(obs.copy())
            total_probe_reward += reward
            if terminated or truncated:
                break

        probe_traces = np.stack(probe_traces, axis=0)  # (N+1, obs_dim)

        if self.ablation == "no_belief":
            # Ran the probe but discard encoding -- z = zeros
            self._z = np.zeros(self.latent_dim, dtype=np.float32)
        else:
            # Full M7: encode probe traces into z
            self._z = self._encode_traces(probe_traces)

        info["probe_steps"] = len(probe_traces) - 1
        info["probe_reward"] = total_probe_reward

        return self._augment_obs(obs), info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info
