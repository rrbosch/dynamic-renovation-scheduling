"""Deep Controlled Learning (DCL) — faithful reproduction.

DCL (Temizöz, Imdahl, Dijkman, Lamghari-Idrissi, van Jaarsveld, EJOR 2025;
arXiv:2011.15122) is approximate policy iteration that casts control as
*classification*. Each of `n` rounds:

  1. collect a fresh ON-POLICY dataset under the current policy π_i (warm-up L
     steps, then step forward labelling each visited state with the
     rollout-IMPROVED action found by a simulation oracle: Sequential-Halving /
     Wilcoxon / fixed budget + Common Random Numbers);
  2. train a CLASSIFIER from scratch on those (state, best-action) pairs → π_{i+1};
  3. π_{i+1} becomes the base/rollout policy for the next round.

Canonical DCL is value-function-free; the deployed policy is just the classifier
(cheap argmax). This module provides the deployed `DCLAgent` (a thin classifier
wrapper) plus the three action-search DECOMPOSITIONS used to adapt DCL to the
combinatorial 4^N joint action space of this problem:

  * ``sequential``  — expanded MDP, decide assets in index order; the classifier
    is conditioned on the partial post-decision state so it anticipates the
    fill-in. Oracle = SequentialMCRolloutAgent.
  * ``independent`` — each asset predicted independently + a feasibility
    coordinator (closest to the plain per-asset classifier). Oracle = local search.
  * ``local_search`` — autoregressive edit policy over a (3N+1) token space
    (set asset i to {repair,renovate,restrict}, or STOP). Oracle = local search.

The approximate-policy-iteration LOOP lives in ``training/dcl_trainer.py``
(DCLTrainer), mirroring PPOAgent/PPOTrainer. The optional truncated-rollout VFA
bootstrap (DCL's opt-in compute shortcut) is owned by the trainer and passed to
the oracle; it is NOT part of the deployed agent.

The previous hybrid implementation is preserved at ``agents/dcl_old.py``.
"""
from __future__ import annotations

import os
import pickle
import numpy as np

from agents.base import Agent
from agents.fn.policy import _asset_features, build_policy_input
from env.mdp import State, InfraEnv, EnvConfig


# ===========================================================================
# Estimators (per-row classifiers shared by all decompositions)
# ===========================================================================

class _XGBEstimator:
    """xgboost multiclass classifier over prepared feature rows. Refit from
    scratch each round (DCL trains the classifier afresh on the new dataset)."""

    DEFAULT_PARAMS = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'n_jobs': -1,
        'verbosity': 0,
    }

    def __init__(self, n_classes: int, params: dict | None = None):
        self.n_classes = int(n_classes)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._clf = None
        self._classes = None       # original integer labels present at fit time
        self._constant = None      # set when a round's labels are a single class
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb
        y = np.asarray(y, dtype=int)
        # xgboost >= 2 dropped the internal label encoder: XGBClassifier needs
        # CONTIGUOUS 0..K-1 targets. Our label sets are sparse (actions may miss a
        # class in a round; the 3N+1 token head is inherently sparse), so encode
        # to dense indices here and map predictions back via self._classes.
        self._classes = np.unique(y)
        if self._classes.size < 2:          # degenerate round → constant predictor
            self._clf, self._constant = None, int(self._classes[0]) if len(y) else 0
            self._fitted = True
            return
        self._constant = None
        enc = np.searchsorted(self._classes, y)
        self._clf = xgb.XGBClassifier(**self.params)
        self._clf.fit(np.asarray(X, dtype=np.float32), enc)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X), dtype=int)
        if self._constant is not None:
            return np.full(len(X), self._constant, dtype=int)
        enc = self._clf.predict(np.asarray(X, dtype=np.float32)).astype(int)
        return self._classes[enc].astype(int)       # map dense index → orig label

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'clf': self._clf, 'fitted': self._fitted,
                         'constant': self._constant, 'classes': self._classes,
                         'n_classes': self.n_classes, 'params': self.params}, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self._clf, self._fitted = d['clf'], d['fitted']
        self._constant = d.get('constant')
        self._classes = d.get('classes')
        self.n_classes, self.params = d['n_classes'], d['params']


class _MLPEstimator:
    """Torch MLP classifier over prepared feature rows. Inputs are z-scored
    (stats frozen on the first fit, like NeuralValueFn / the PPO nets — essential
    for conditioning); trained multi-epoch with cross-entropy each round."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dims=None,
                 lr: float = 1e-3, epochs: int = 30, batch_size: int = 256):
        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.hidden_dims = list(hidden_dims) if hidden_dims else [256, 256]
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self._model = None
        self._mu = None
        self._sd = None
        self._fitted = False

    def _build(self):
        import torch.nn as nn
        layers, d = [], self.in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, self.n_classes))
        self._model = nn.Sequential(*layers).float()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn.functional as F
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        # Faithful DCL retrains the classifier FROM SCRATCH each round: re-init the
        # weights and recompute input normalization on this round's fresh dataset
        # (mirrors _XGBEstimator, which builds a new XGBClassifier each fit).
        self._build()
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0) + 1e-6
        Xn = torch.tensor((X - self._mu) / self._sd, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        n = len(Xn)
        self._model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for b in range(0, n, self.batch_size):
                idx = perm[b:b + self.batch_size]
                logits = self._model(Xn[idx])
                loss = F.cross_entropy(logits, Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X), dtype=int)
        import torch
        Xn = (np.asarray(X, dtype=np.float32) - self._mu) / self._sd
        with torch.no_grad():
            logits = self._model(torch.tensor(Xn, dtype=torch.float32))
            return logits.argmax(dim=-1).numpy().astype(int)

    def save(self, path: str) -> None:
        import torch
        torch.save({'state_dict': (self._model.state_dict()
                                   if self._model is not None else None),
                    'mu': self._mu, 'sd': self._sd, 'fitted': self._fitted,
                    'in_dim': self.in_dim, 'n_classes': self.n_classes,
                    'hidden_dims': self.hidden_dims}, path)

    def load(self, path: str) -> None:
        import torch
        d = torch.load(path, weights_only=False)
        self.in_dim, self.n_classes = d['in_dim'], d['n_classes']
        self.hidden_dims = d['hidden_dims']
        self._build()
        if d['state_dict'] is not None:
            self._model.load_state_dict(d['state_dict'])
        self._mu, self._sd, self._fitted = d['mu'], d['sd'], d['fitted']


def _make_estimator(kind: str, in_dim: int, n_classes: int, *,
                    hidden_dims=None, lr=1e-3, epochs=30, batch_size=256,
                    clf_kwargs=None):
    if kind == 'xgboost':
        return _XGBEstimator(n_classes=n_classes, params=clf_kwargs)
    if kind == 'nn':
        return _MLPEstimator(in_dim=in_dim, n_classes=n_classes,
                             hidden_dims=hidden_dims, lr=lr,
                             epochs=epochs, batch_size=batch_size)
    raise ValueError(f"Unknown estimator kind: {kind!r}")


# ===========================================================================
# Decompositions
# ===========================================================================

class _BaseDecomposition:
    """Common interface. A decomposition is BOTH the deployed policy (act/fit/
    save/load/_fitted) and the factory for the labelling oracle."""

    name = 'base'

    def __init__(self, env: InfraEnv, estimator_kind: str = 'xgboost',
                 use_global_context: bool = True, hidden_dims=None, lr=1e-3,
                 epochs=30, batch_size=256, clf_kwargs=None):
        self.env = env
        self.cfg = env.config
        self.estimator_kind = estimator_kind
        self.use_global_context = use_global_context
        self._est_kwargs = dict(hidden_dims=hidden_dims, lr=lr, epochs=epochs,
                                batch_size=batch_size, clf_kwargs=clf_kwargs)
        self.est = None  # built lazily once in/out dims are known

    @property
    def _fitted(self) -> bool:
        return self.est is not None and self.est._fitted

    # -- oracle ------------------------------------------------------------
    def make_oracle(self, base_policy, dcl_cfg, seed: int, value_fn=None):
        raise NotImplementedError

    def _oracle_kwargs(self, base_policy, dcl_cfg, seed, value_fn):
        sel = 'adaptive' if dcl_cfg.rollout_selection == 'wilcoxon' else dcl_cfg.rollout_selection
        return dict(
            rollout_policy=base_policy, env=self.env,
            n_rollouts=dcl_cfg.n_rollouts, rollout_horizon=dcl_cfg.rollout_horizon,
            seed=seed, action_threshold=dcl_cfg.action_threshold,
            initial_action=dcl_cfg.initial_action, selection=sel,
            p_threshold=dcl_cfg.p_threshold, min_rollouts=dcl_cfg.min_rollouts,
            max_rollouts=dcl_cfg.max_rollouts, rollout_batch=dcl_cfg.rollout_batch,
            value_fn=value_fn, sh_budget_per_arm=dcl_cfg.sh_budget_per_arm,
        )

    # -- dataset / training -----------------------------------------------
    def build_rows(self, state: State, label: np.ndarray):
        """Yield (feature_row, target_class) tuples for one labelled state."""
        raise NotImplementedError

    def fit(self, states: list[State], labels: np.ndarray) -> None:
        X, y = [], []
        for s, lab in zip(states, labels):
            for feat, tgt in self.build_rows(s, lab):
                X.append(feat)
                y.append(tgt)
        self.est.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int))

    # -- deployment --------------------------------------------------------
    def act(self, state: State) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str) -> None:
        if self.est is not None:
            self.est.save(path)

    def load(self, path: str) -> None:
        if self.est is not None and os.path.exists(path):
            self.est.load(path)


class SequentialDecomposition(_BaseDecomposition):
    """Per-asset expanded MDP. Assets decided in index order; the classifier is
    conditioned on the PARTIAL post-decision state (committed prefix applied), so
    it anticipates how later assets are filled in. Oracle = SequentialMCRollout."""

    name = 'sequential'

    def __init__(self, env, **kw):
        super().__init__(env, **kw)
        F = _asset_features(_dummy_state(self.cfg), self.cfg,
                            self.use_global_context).shape[1]
        # +4 commit-context features (see _ctx) so the classifier can anticipate
        # how the joint action is being filled in (the per-asset _asset_features
        # global context alone carries no count of already-committed actions).
        self.est = _make_estimator(self.estimator_kind, in_dim=F + 4, n_classes=4,
                                   **self._est_kwargs)

    def _ctx(self, action: np.ndarray, i: int) -> np.ndarray:
        """Commit-context for the i-th decision: fraction of assets already
        committed, and fractions committed to repair / renovate / restrict
        (computed over the prefix 0..i-1). Lets the per-asset classifier see the
        in-progress fill-in (budget / concurrency / network coupling)."""
        N = self.cfg.n_assets
        pref = action[:i]
        return np.array([
            i / N,
            float(np.sum(pref == InfraEnv.ACTION_REPAIR)) / N,
            float(np.sum(pref == InfraEnv.ACTION_RENOVATE)) / N,
            float(np.sum(pref == InfraEnv.ACTION_RESTRICT)) / N,
        ], dtype=np.float32)

    def make_oracle(self, base_policy, dcl_cfg, seed, value_fn=None):
        from agents.rollout import SequentialMCRolloutAgent
        return SequentialMCRolloutAgent(**self._oracle_kwargs(
            base_policy, dcl_cfg, seed, value_fn))

    def build_rows(self, state, label):
        N = self.cfg.n_assets
        action = np.zeros(N, dtype=int)
        for i in range(N):
            partial = self.env.post_decision_state(state, action, check=False)
            feats = _asset_features(partial, self.cfg, self.use_global_context)
            yield np.concatenate([feats[i], self._ctx(action, i)]), int(label[i])
            action[i] = int(label[i])           # commit the label and advance

    def act(self, state):
        N = self.cfg.n_assets
        feas = self.env.feasible_actions(state)
        action = np.zeros(N, dtype=int)
        for i in range(N):
            partial = self.env.post_decision_state(state, action, check=False)
            feats = _asset_features(partial, self.cfg, self.use_global_context)
            row = np.concatenate([feats[i], self._ctx(action, i)])[None, :]
            a = int(self.est.predict(row)[0])
            if not feas[i, a]:
                a = InfraEnv.ACTION_NONE
            action[i] = a
        return action


class IndependentDecomposition(_BaseDecomposition):
    """Each asset predicted independently from the (pre-decision) state, then a
    feasibility coordinator projects infeasible picks to ACTION_NONE. Oracle =
    local-search MC rollout."""

    name = 'independent'

    def __init__(self, env, **kw):
        super().__init__(env, **kw)
        F = _asset_features(_dummy_state(self.cfg), self.cfg,
                            self.use_global_context).shape[1]
        self.est = _make_estimator(self.estimator_kind, in_dim=F, n_classes=4,
                                   **self._est_kwargs)

    def make_oracle(self, base_policy, dcl_cfg, seed, value_fn=None):
        from agents.rollout import MonteCarloRolloutAgent
        return MonteCarloRolloutAgent(**self._oracle_kwargs(
            base_policy, dcl_cfg, seed, value_fn))

    def build_rows(self, state, label):
        feats = _asset_features(state, self.cfg, self.use_global_context)
        for i in range(self.cfg.n_assets):
            yield feats[i], int(label[i])

    def act(self, state):
        feats = _asset_features(state, self.cfg, self.use_global_context)
        pred = self.est.predict(feats)                      # (N,)
        feas = self.env.feasible_actions(state)             # coordinator
        N = self.cfg.n_assets
        return np.where(feas[np.arange(N), pred], pred, InfraEnv.ACTION_NONE)


class LocalSearchStopDecomposition(_BaseDecomposition):
    """Autoregressive edit policy. Token space = 3N + 1: token 3*i + (a-1) sets
    asset i to action a∈{repair,renovate,restrict}; token 3N = STOP. The
    classifier maps (flat state features, current partial-action one-hot) → next
    token; at deployment edits are applied until STOP. Oracle = local search,
    whose final joint action is decomposed into a canonical edit sequence as the
    training labels."""

    name = 'local_search'

    def __init__(self, env, finite_horizon: bool = True, **kw):
        super().__init__(env, **kw)
        self.finite_horizon = finite_horizon
        N = self.cfg.n_assets
        self.stop_token = 3 * N
        n_classes = 3 * N + 1
        state_dim = 5 * N + (1 if finite_horizon else 0)
        in_dim = state_dim + 4 * N                          # state + action one-hot
        self.est = _make_estimator(self.estimator_kind, in_dim=in_dim,
                                   n_classes=n_classes, **self._est_kwargs)

    def make_oracle(self, base_policy, dcl_cfg, seed, value_fn=None):
        from agents.rollout import MonteCarloRolloutAgent
        return MonteCarloRolloutAgent(**self._oracle_kwargs(
            base_policy, dcl_cfg, seed, value_fn))

    def _feat(self, state, action):
        N = self.cfg.n_assets
        sfeat = build_policy_input(state, N, self.cfg.T, self.finite_horizon)
        onehot = np.zeros(4 * N, dtype=np.float32)
        onehot[np.arange(N) * 4 + action] = 1.0
        return np.concatenate([sfeat, onehot]).astype(np.float32)

    def build_rows(self, state, label):
        N = self.cfg.n_assets
        action = np.zeros(N, dtype=int)
        for i in range(N):
            a = int(label[i])
            if a == InfraEnv.ACTION_NONE:
                continue
            yield self._feat(state, action), 3 * i + (a - 1)
            action[i] = a
        yield self._feat(state, action), self.stop_token        # STOP

    def act(self, state):
        N = self.cfg.n_assets
        feas = self.env.feasible_actions(state)
        action = np.zeros(N, dtype=int)
        for _ in range(N):                                   # at most N edits
            tok = int(self.est.predict(self._feat(state, action)[None, :])[0])
            if tok == self.stop_token:
                break
            i, a = divmod(tok, 3)
            a += 1
            if i >= N or not feas[i, a]:
                break                                        # stop on infeasible
            action[i] = a
        return action


_DECOMPOSITIONS = {
    'sequential': SequentialDecomposition,
    'independent': IndependentDecomposition,
    'local_search': LocalSearchStopDecomposition,
}


def build_decomposition(action_search: str, env: InfraEnv, **kw) -> _BaseDecomposition:
    if action_search not in _DECOMPOSITIONS:
        raise ValueError(
            f"Unknown action_search {action_search!r}; "
            f"valid: {sorted(_DECOMPOSITIONS)}")
    cls = _DECOMPOSITIONS[action_search]
    if action_search != 'local_search':
        kw.pop('finite_horizon', None)            # only the STOP head uses it
    return cls(env, **kw)


def _dummy_state(cfg: EnvConfig) -> State:
    N = cfg.n_assets
    z = np.zeros(N, dtype=float)
    s = State(z.copy(), z.copy(), z.copy(), z.copy(), z.copy())
    s.t = 0
    return s


# ===========================================================================
# Deployed agent
# ===========================================================================

class DCLAgent(Agent):
    """Deployed DCL policy. ``act`` runs the trained classifier (the decomposition
    policy); before any policy-iteration round has trained it, it falls back to
    the base heuristic. The rollout-improvement oracle and the round loop live in
    ``DCLTrainer`` — this agent is a thin wrapper, like PPOAgent."""

    def __init__(self, policy: _BaseDecomposition, base_heuristic: Agent,
                 env: InfraEnv, action_search: str = 'sequential'):
        self.policy = policy
        self.base_heuristic = base_heuristic
        self.env = env
        self.action_search = action_search
        self.step_metrics: dict = {}

    def act(self, state: State) -> np.ndarray:
        if self.policy._fitted:
            return self.policy.act(state)
        return self.base_heuristic.act(state)

    # The round loop is driven by DCLTrainer; update() is unused here but kept so
    # the agent is a "learner" by the Agent ABC convention if ever routed through
    # the generic Trainer.
    def update(self, transitions: list) -> None:  # pragma: no cover - unused
        pass

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.policy.save(os.path.join(path, 'policy.bin'))

    def load(self, path: str) -> None:
        self.policy.load(os.path.join(path, 'policy.bin'))
