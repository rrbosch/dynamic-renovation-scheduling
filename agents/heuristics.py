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


class DoNothingAgent(HeuristicAgent):
    """Always do nothing (all-zero action).

    A behavior baseline; primarily used as one mode of the warmstart ``MixtureAgent``,
    where sustained do-nothing episodes are the only way to build up the high-``n_fail``
    (escalating-risk) states a reactive warmstart never reaches — the coverage fix for
    the ADP value function's failure-cost under-prediction
    (see docs/adp_value_fn_improvements.md).
    """

    def __init__(self, env_config: EnvConfig):
        self.env_config = env_config

    def act(self, state: State) -> np.ndarray:
        return np.zeros(self.env_config.n_assets, dtype=int)

    def _heuristic_params(self) -> dict:
        return {}


class FlipWrapperAgent(Agent):
    """Wrap a base policy with the warmstart "bit-flip" exploration.

    Per asset, with a condition(d)-ramped probability `p_flip` (from `p_base` at
    `d<=d_ref` up to `p_high` at `d=d_fail`), override the base action with a random
    *feasible* action biased toward acting (esp. renovate). This gives the value
    function coverage of acting post-states near failure that a ~98%-do-nothing
    heuristic never produces.

    Previously baked into `Trainer._run_warmstart`; now a composable behavior wrapper
    (a warmstart `MixtureAgent` can flip-wrap its reactive modes while leaving the
    do-nothing mode pure). COLLECTION-ONLY / not parallelism-invariant: it consumes an
    explicit *per-step* `np.random.Generator`, deterministic for the sequential
    warmstart loop (same family as `MixtureAgent`). With `p_base == p_high == 0` it is a
    transparent pass-through of `base`.
    """

    ACT_NONE, ACT_REPAIR, ACT_RENOVATE, ACT_RESTRICT = 0, 1, 2, 3

    def __init__(self, base: Agent, env: InfraEnv, rng: np.random.Generator,
                 p_base: float = 1.0 / 120, p_high: float = 0.50, d_ref: float = 0.5,
                 act_bias: float = 0.9, renovate_bias: float = 0.7):
        self.base = base
        self.env = env
        self._rng = rng
        self.p_base = p_base
        self.p_high = p_high
        self.d_ref = d_ref
        self.act_bias = act_bias
        self.renovate_bias = renovate_bias
        self._denom = max(1e-9, float(env.config.d_fail) - d_ref)

    def act(self, state: State) -> np.ndarray:
        action = np.asarray(self.base.act(state)).copy()
        if self.p_base == 0.0 and self.p_high == 0.0:
            return action
        rng = self._rng
        feas = self.env.feasible_actions(state)            # (N, 4) bool
        ramp = np.clip((state.d - self.d_ref) / self._denom, 0.0, 1.0)
        p_flip_i = self.p_base + (self.p_high - self.p_base) * ramp
        for i in range(self.env.config.n_assets):
            if rng.random() >= p_flip_i[i]:
                continue
            feasible_i = np.where(feas[i])[0]
            acting = feasible_i[feasible_i != self.ACT_NONE]
            if acting.size > 0 and rng.random() < self.act_bias:
                if self.ACT_RENOVATE in acting and rng.random() < self.renovate_bias:
                    action[i] = self.ACT_RENOVATE
                else:
                    rest = acting[acting != self.ACT_RENOVATE]
                    action[i] = int(rng.choice(rest if rest.size > 0 else acting))
            else:
                action[i] = int(rng.choice(feasible_i))
        return action


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


# ---------------------------------------------------------------------------
# Shared helpers for the network-/lifetime-aware heuristics below
# ---------------------------------------------------------------------------

def _remaining_life_epochs(state: State, cfg: EnvConfig) -> np.ndarray:
    """Expected epochs until condition reaches the failure threshold.

    Uses the same degradation model as env.degradation.gamma_step: the per-epoch
    mean increment is ``E[ΔG] = alpha_eff * dt / beta`` with
    ``alpha_eff = (1 - (1 - restrict_degrad_multiplier) * ell) * alpha0``. Assets
    with a zero mean rate get +inf (never failing). Result shape (N,).
    """
    f = cfg.restrict_degrad_multiplier
    alpha_eff = (1.0 - (1.0 - f) * state.ell) * cfg.alpha0
    mean_inc = alpha_eff * cfg.dt / cfg.beta              # per epoch
    d_remaining = np.maximum(0.0, cfg.d_fail - state.d)
    return np.where(mean_inc > 1e-12, d_remaining / mean_inc, np.inf)


def _select_top_k(eligible: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the (up to) k highest-scoring eligible assets.

    Ties and non-eligible assets are handled by masking scores to -inf. k is
    clamped to [0, n_eligible]. Returns a bool mask shape (N,).
    """
    n = eligible.shape[0]
    mask = np.zeros(n, dtype=bool)
    k = int(max(0, k))
    if k == 0:
        return mask
    masked = np.where(eligible, score, -np.inf)
    n_elig = int(eligible.sum())
    k = min(k, n_elig)
    if k == 0:
        return mask
    # argpartition picks the k largest without a full sort
    idx = np.argpartition(masked, n - k)[n - k:]
    idx = idx[eligible[idx]]            # drop any -inf fillers if n_elig < k
    mask[idx] = True
    return mask


class LeadTimeAgent(HeuristicAgent):
    """
    Predictive (just-in-time) renovation keyed on expected remaining life rather
    than a fixed condition threshold.

    For each asset, R_i = expected epochs until d_i reaches d_fail (see
    _remaining_life_epochs). Because alpha0/beta are heterogeneous, a fixed d
    threshold is a poor proxy for "about to fail"; this triggers off R_i instead.

    Priority (highest first): renovate > repair > restrict > nothing.
    - Renovate if R_i <= lead_epochs (or already failed), and h_i <= 0.
    - Repair   if repair_lead is not None and R_i <= repair_lead, r_i == 0.
    - Restrict if restrict_lead is not None and R_i <= restrict_lead, ell_i == 0.
    """

    def __init__(
        self,
        lead_epochs: float,
        env_config: EnvConfig,
        repair_lead: float | None = None,
        restrict_lead: float | None = None,
    ):
        self.lead_epochs = lead_epochs
        self.repair_lead = repair_lead
        self.restrict_lead = restrict_lead
        self.env_config = env_config

    def act(self, state: State) -> np.ndarray:
        cfg = self.env_config
        n = cfg.n_assets
        action = np.zeros(n, dtype=int)
        eligible = state.h <= 0
        R = _remaining_life_epochs(state, cfg)

        # Priority 3 (lowest): restrict
        if self.restrict_lead is not None and cfg.allow_restrict:
            restrict_mask = eligible & (R <= self.restrict_lead) & (state.ell == 0)
            action[restrict_mask] = InfraEnv.ACTION_RESTRICT

        # Priority 2: repair (overwrites restrict)
        if self.repair_lead is not None and cfg.allow_repair:
            repair_mask = eligible & (R <= self.repair_lead) & (state.r == 0)
            action[repair_mask] = InfraEnv.ACTION_REPAIR

        # Priority 1 (highest): renovate — lead-time trigger or already failed.
        renovate_mask = eligible & ((R <= self.lead_epochs) | (state.d >= cfg.d_fail))
        action[renovate_mask] = InfraEnv.ACTION_RENOVATE

        return action

    def _heuristic_params(self) -> dict:
        return {
            'lead_epochs': self.lead_epochs,
            'repair_lead': self.repair_lead,
            'restrict_lead': self.restrict_lead,
        }


class NetConcurrencyAgent(HeuristicAgent):
    """
    Network-aware, concurrency-limited renovation.

    A reactive d-threshold flags renovation candidates, but only `max_concurrent`
    renovations may run at once (counting in-progress h>0). When slots are scarce
    the policy *spreads* the travel-cost impact: candidate priority is
    ``d_i - spread_penalty * normalized_flow_i`` so busy (high nominal-flow) edges
    — which are the most expensive to drop to eta_ren capacity — are deferred in
    favour of low-flow/redundant edges, until they are forced.

    Failed assets (d >= d_fail) are renovated unconditionally (emergency), even if
    that temporarily exceeds max_concurrent.

    `asset_flow` is a precomputed (N,) nominal-capacity TAP flow per asset edge,
    supplied at construction by the experiment builder (never recomputed in act()).
    If None, the policy degrades gracefully to a pure worst-first concurrency cap.
    """

    def __init__(
        self,
        threshold: float,
        env_config: EnvConfig,
        max_concurrent: int = 3,
        spread_penalty: float = 0.0,
        asset_flow: np.ndarray | None = None,
    ):
        self.threshold = threshold
        self.max_concurrent = int(max_concurrent)
        self.spread_penalty = spread_penalty
        self.env_config = env_config
        self._set_flow(asset_flow)

    def _set_flow(self, asset_flow: np.ndarray | None) -> None:
        if asset_flow is None:
            self._norm_flow = np.zeros(self.env_config.n_assets)
        else:
            af = np.asarray(asset_flow, dtype=float)
            self._norm_flow = af / max(float(af.max()), 1e-9)

    def act(self, state: State) -> np.ndarray:
        cfg = self.env_config
        n = cfg.n_assets
        action = np.zeros(n, dtype=int)
        eligible = state.h <= 0

        # Emergency: renovate every eligible failed asset, no matter the budget.
        forced = eligible & (state.d >= cfg.d_fail)
        action[forced] = InfraEnv.ACTION_RENOVATE

        in_progress = int(np.sum(state.h > 0))
        used = in_progress + int(forced.sum())
        slots = self.max_concurrent - used
        if slots <= 0:
            return action

        # Remaining candidates ranked by urgency minus a flow (congestion) penalty.
        candidates = eligible & (state.d >= self.threshold) & ~forced
        priority = state.d - self.spread_penalty * self._norm_flow
        chosen = _select_top_k(candidates, priority, slots)
        action[chosen] = InfraEnv.ACTION_RENOVATE
        return action

    def _heuristic_params(self) -> dict:
        return {
            'threshold': self.threshold,
            'max_concurrent': self.max_concurrent,
            'spread_penalty': self.spread_penalty,
        }

    def _apply_params(self, p: dict) -> None:
        self.threshold = p['threshold']
        self.max_concurrent = int(p['max_concurrent'])
        self.spread_penalty = p['spread_penalty']


class HoldingAgent(HeuristicAgent):
    """
    Concurrency-capped renovation with an explicit restrict/repair "holding" layer.

    Renovation is reactive (d >= threshold) and capped at `max_concurrent`
    (in-progress + new, failed assets forced). Assets that are in danger
    (remaining life R_i <= defer_window) but cannot get a renovation slot are held:
      - restrict if the asset sits on a low-flow/redundant edge (asset_flow below
        the `restrict_flow_quantile` quantile) — slows degradation cheaply, since
        the capacity hit barely matters on a redundant edge;
      - else repair (if r_i == 0) — a cheap one-shot condition bump.

    Priority (highest first): renovate > repair > restrict > nothing.
    """

    def __init__(
        self,
        threshold: float,
        env_config: EnvConfig,
        max_concurrent: int = 3,
        defer_window: float = 4.0,
        restrict_flow_quantile: float = 0.5,
        asset_flow: np.ndarray | None = None,
    ):
        self.threshold = threshold
        self.max_concurrent = int(max_concurrent)
        self.defer_window = defer_window
        self.restrict_flow_quantile = restrict_flow_quantile
        self.env_config = env_config
        self._set_flow(asset_flow)

    def _set_flow(self, asset_flow: np.ndarray | None) -> None:
        n = self.env_config.n_assets
        self._asset_flow = (np.zeros(n) if asset_flow is None
                            else np.asarray(asset_flow, dtype=float))

    def act(self, state: State) -> np.ndarray:
        cfg = self.env_config
        n = cfg.n_assets
        action = np.zeros(n, dtype=int)
        eligible = state.h <= 0
        R = _remaining_life_epochs(state, cfg)

        # --- Holding layer (lowest priority; written first, overwritten below) ---
        in_danger = eligible & (R <= self.defer_window)
        low_flow_cut = np.quantile(self._asset_flow, self.restrict_flow_quantile)
        is_low_flow = self._asset_flow <= low_flow_cut

        if cfg.allow_restrict:
            restrict_mask = in_danger & is_low_flow & (state.ell == 0)
            action[restrict_mask] = InfraEnv.ACTION_RESTRICT
        if cfg.allow_repair:
            repair_mask = in_danger & ~is_low_flow & (state.r == 0)
            action[repair_mask] = InfraEnv.ACTION_REPAIR

        # --- Renovation layer (highest priority; overwrites holding actions) ---
        forced = eligible & (state.d >= cfg.d_fail)
        action[forced] = InfraEnv.ACTION_RENOVATE

        in_progress = int(np.sum(state.h > 0))
        slots = self.max_concurrent - in_progress - int(forced.sum())
        candidates = eligible & (state.d >= self.threshold) & ~forced
        if slots > 0:
            # Most urgent first: lowest remaining life.
            chosen = _select_top_k(candidates, -R, slots)
            action[chosen] = InfraEnv.ACTION_RENOVATE

        return action

    def _heuristic_params(self) -> dict:
        return {
            'threshold': self.threshold,
            'max_concurrent': self.max_concurrent,
            'defer_window': self.defer_window,
            'restrict_flow_quantile': self.restrict_flow_quantile,
        }

    def _apply_params(self, p: dict) -> None:
        self.threshold = p['threshold']
        self.max_concurrent = int(p['max_concurrent'])
        self.defer_window = p['defer_window']
        self.restrict_flow_quantile = p['restrict_flow_quantile']


class ValueDensityAgent(HeuristicAgent):
    """
    Bang-per-buck greedy renovation.

    Ranks eligible assets by a cost-density score and renovates the top
    `max_concurrent` (counting in-progress h>0; failed assets forced):

        score_i = (risk_weight * n_fail_i * L_i * risk_base * dt
                   + degrad_weight * d_i * L_i) / c_ren_i

    The first term is the risk cost currently being paid; the second is a proxy
    for imminent degradation cost (proximity-to-failure × size). Dividing by the
    renovation cost gives euros-avoided-per-euro-spent. Optional `threshold`
    excludes assets below a condition floor from being candidates.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        max_concurrent: int = 3,
        risk_weight: float = 1.0,
        degrad_weight: float = 1.0,
        threshold: float = 0.0,
    ):
        self.env_config = env_config
        self.max_concurrent = int(max_concurrent)
        self.risk_weight = risk_weight
        self.degrad_weight = degrad_weight
        self.threshold = threshold

    def act(self, state: State) -> np.ndarray:
        cfg = self.env_config
        n = cfg.n_assets
        action = np.zeros(n, dtype=int)
        eligible = state.h <= 0

        forced = eligible & (state.d >= cfg.d_fail)
        action[forced] = InfraEnv.ACTION_RENOVATE

        in_progress = int(np.sum(state.h > 0))
        slots = self.max_concurrent - in_progress - int(forced.sum())
        if slots <= 0:
            return action

        L = cfg.asset_lengths_m
        risk_term = self.risk_weight * state.n_fail * L * cfg.risk_base * cfg.dt
        degrad_term = self.degrad_weight * state.d * L
        score = (risk_term + degrad_term) / np.maximum(cfg.c_ren, 1e-9)

        candidates = eligible & (state.d >= self.threshold) & ~forced
        chosen = _select_top_k(candidates, score, slots)
        action[chosen] = InfraEnv.ACTION_RENOVATE
        return action

    def _heuristic_params(self) -> dict:
        return {
            'max_concurrent': self.max_concurrent,
            'risk_weight': self.risk_weight,
            'degrad_weight': self.degrad_weight,
            'threshold': self.threshold,
        }

    def _apply_params(self, p: dict) -> None:
        self.max_concurrent = int(p['max_concurrent'])
        self.risk_weight = p['risk_weight']
        self.degrad_weight = p['degrad_weight']
        self.threshold = p['threshold']


class WorstFirstAgent(HeuristicAgent):
    """
    Classic worst-first baseline: renovate the most-degraded eligible assets
    (optionally length-weighted, d_i * L_i) up to `max_concurrent` concurrent
    renovations. Assets at/above a condition `threshold` are candidates; failed
    assets are forced.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        max_concurrent: int = 3,
        threshold: float = 0.5,
        use_length: bool = True,
    ):
        self.env_config = env_config
        self.max_concurrent = int(max_concurrent)
        self.threshold = threshold
        self.use_length = bool(use_length)

    def act(self, state: State) -> np.ndarray:
        cfg = self.env_config
        n = cfg.n_assets
        action = np.zeros(n, dtype=int)
        eligible = state.h <= 0

        forced = eligible & (state.d >= cfg.d_fail)
        action[forced] = InfraEnv.ACTION_RENOVATE

        in_progress = int(np.sum(state.h > 0))
        slots = self.max_concurrent - in_progress - int(forced.sum())
        if slots <= 0:
            return action

        score = state.d * (cfg.asset_lengths_m if self.use_length else 1.0)
        candidates = eligible & (state.d >= self.threshold) & ~forced
        chosen = _select_top_k(candidates, score, slots)
        action[chosen] = InfraEnv.ACTION_RENOVATE
        return action

    def _heuristic_params(self) -> dict:
        return {
            'max_concurrent': self.max_concurrent,
            'threshold': self.threshold,
            'use_length': self.use_length,
        }

    def _apply_params(self, p: dict) -> None:
        self.max_concurrent = int(p['max_concurrent'])
        self.threshold = p['threshold']
        self.use_length = bool(p['use_length'])
