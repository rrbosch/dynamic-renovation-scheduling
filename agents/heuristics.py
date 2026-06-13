"""Heuristic agents: Reactive and Paced."""
from __future__ import annotations

import json
import os

import numpy as np

from agents.base import Agent
from env.mdp import State, EnvConfig, InfraEnv


class HeuristicAgent(Agent):
    """
    Intermediate base for parameter-based heuristic agents.
    Overrides Agent.save/load with a JSON params file.
    Subclasses implement _heuristic_params() and optionally _apply_params().
    """

    def _heuristic_params(self) -> dict:
        """Return JSON-serializable constructor parameters."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'params.json'), 'w') as f:
            json.dump(self._heuristic_params(), f, indent=2)

    def load(self, path: str) -> None:
        """Restore parameters in-place from params.json."""
        with open(os.path.join(path, 'params.json')) as f:
            p = json.load(f)
        self._apply_params(p)

    def _apply_params(self, p: dict) -> None:
        """Default: setattr for each key. Override for non-scalar params."""
        for k, v in p.items():
            setattr(self, k, v)


class ReactiveAgent(HeuristicAgent):
    """
    Per-asset action priority (highest to lowest): renovate > repair > restrict > nothing.

    - Renovate if d_i >= threshold and h_i <= 0.
    - Repair   if repair_threshold is not None, d_i >= repair_threshold, h_i <= 0,
               not already scheduled for renovation, and r_i == 0 (unused this cycle).
    - Restrict if restrict_threshold is not None, d_i >= restrict_threshold, h_i <= 0,
               not already scheduled for renovation or repair, and ell_i == 0.
    """

    def __init__(
        self,
        threshold: float,
        env_config: EnvConfig,
        repair_threshold: float | None = None,
        restrict_threshold: float | None = None,
    ):
        self.threshold = threshold
        self.repair_threshold = repair_threshold
        self.restrict_threshold = restrict_threshold
        self.env_config = env_config

    def act(self, state: State) -> np.ndarray:
        n = self.env_config.n_assets
        action = np.zeros(n, dtype=int)
        eligible = state.h <= 0  # not currently under renovation

        # Priority 3 (lowest): restrict
        if self.restrict_threshold is not None:
            restrict_mask = (
                eligible
                & (state.d >= self.restrict_threshold)
                & (state.ell == 0)
            )
            action[restrict_mask] = InfraEnv.ACTION_RESTRICT

        # Priority 2: repair (overwrites restrict)
        if self.repair_threshold is not None:
            repair_mask = (
                eligible
                & (state.d >= self.repair_threshold)
                & (state.r == 0)
            )
            action[repair_mask] = InfraEnv.ACTION_REPAIR

        # Priority 1 (highest): renovate (overwrites repair and restrict)
        renovate_mask = eligible & (state.d >= self.threshold)
        action[renovate_mask] = InfraEnv.ACTION_RENOVATE

        return action

    def _heuristic_params(self) -> dict:
        return {
            'threshold': self.threshold,
            'repair_threshold': self.repair_threshold,
            'restrict_threshold': self.restrict_threshold,
        }


class PacedAgent(HeuristicAgent):
    """
    Initiates renovations to match the required pace based on expected lifespans.
    """

    def __init__(self, threshold: float, env_config: EnvConfig, pace_threshold: float = 0.5):
        self.threshold = threshold
        self.env_config = env_config
        self.pace_threshold = pace_threshold

    def act(self, state: State) -> np.ndarray:
        cfg = self.env_config
        n = cfg.n_assets

        # Expected remaining lifespan per asset
        # R_i = (d_thr - d_i) / E[degradation rate]
        # E[rate] = alpha_i(ell_i) / beta_i
        alpha_eff = (1.0 - 0.5 * state.ell) * cfg.alpha0
        mean_rate = alpha_eff / cfg.beta  # per epoch
        d_remaining = np.maximum(0.0, self.threshold - state.d)
        # Avoid division by zero
        R = np.where(mean_rate > 1e-12, d_remaining / mean_rate, np.inf)

        # Required pace: N / sum(R_i)  (renovations per epoch)
        finite_R = R[np.isfinite(R)]
        if len(finite_R) == 0 or finite_R.sum() == 0:
            return np.zeros(n, dtype=int)

        mu_h = np.broadcast_to(np.asarray(cfg.mu_h, dtype=float), n)
        total_ren_duration = np.sum(1.0 / (mu_h * cfg.dt))
        required_pace = total_ren_duration / finite_R.sum()

        # Sort eligible assets by lowest R_i first
        eligible = state.h <= 0
        priorities = np.where(eligible, R, np.inf)
        sorted_idx = np.argsort(priorities)

        action = np.zeros(n, dtype=int)
        current_renovating = int(np.sum(state.h > 0))

        # Safeguard: force-renovate any eligible asset at or above threshold
        forced = eligible & (state.d >= self.threshold)
        action[forced] = InfraEnv.ACTION_RENOVATE
        current_renovating += int(np.sum(forced))

        for i in sorted_idx:
            if not eligible[i] or forced[i]:
                continue
            if required_pace - current_renovating / max(n, 1) < self.pace_threshold:
                break
            action[i] = InfraEnv.ACTION_RENOVATE
            current_renovating += 1

        return action

    def _heuristic_params(self) -> dict:
        return {'threshold': self.threshold, 'pace_threshold': self.pace_threshold}


class PerAssetReactiveAgent(HeuristicAgent):
    """
    Reactive policy with independent thresholds per asset.

    thresholds: array shape (N, 3)
        [:, 0] = repair_threshold
        [:, 1] = restrict_threshold
        [:, 2] = renovate_threshold
    Priority (highest): renovate > repair > restrict > nothing.
    Threshold >= 1.0 effectively disables that action.
    """

    def __init__(self, thresholds: np.ndarray, env_config: EnvConfig):
        self._thr = np.asarray(thresholds, dtype=float)  # (N, 3)
        self._cfg = env_config

    @classmethod
    def from_file(cls, path: str, env_config: EnvConfig) -> 'PerAssetReactiveAgent':
        """Load thresholds from a ga_thresholds.json file saved by GeneticAlgorithmAgent."""
        with open(path) as f:
            data = json.load(f)
        thr = np.column_stack([
            data['repair_threshold'],
            data['restrict_threshold'],
            data['renovate_threshold'],
        ])  # shape (N, 3)
        return cls(thr, env_config)

    def act(self, state: State) -> np.ndarray:
        n = self._cfg.n_assets
        action = np.zeros(n, dtype=int)
        eligible = state.h <= 0  # not under renovation

        rep_thr = self._thr[:, 0]
        res_thr = self._thr[:, 1]
        ren_thr = self._thr[:, 2]

        # Priority 3 (lowest): restrict
        restrict_mask = eligible & (state.d >= res_thr) & (state.ell == 0)
        action[restrict_mask] = InfraEnv.ACTION_RESTRICT

        # Priority 2: repair (overwrites restrict)
        repair_mask = eligible & (state.d >= rep_thr) & (state.r == 0)
        action[repair_mask] = InfraEnv.ACTION_REPAIR

        # Priority 1 (highest): renovate (overwrites repair and restrict)
        renovate_mask = eligible & (state.d >= ren_thr)
        action[renovate_mask] = InfraEnv.ACTION_RENOVATE

        return action

    def _heuristic_params(self) -> dict:
        return {'thresholds': self._thr.tolist()}

    def _apply_params(self, p: dict) -> None:
        self._thr = np.array(p['thresholds'])
