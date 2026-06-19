"""Policy network and policy classes."""
from __future__ import annotations

import pickle
import numpy as np

from env.mdp import State, EnvConfig


# ---------------------------------------------------------------------------
# PolicyNetwork
# ---------------------------------------------------------------------------

class PolicyNetwork:
    """
    MLP in torch: input → hidden → hidden → (N × 4) logits.
    """

    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        n_actions: int = 4,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        T: int | None = None,
        finite_horizon: bool = True,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.input_dim = input_dim
        self.n_assets = n_assets
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        self.lr = lr
        # T / finite_horizon drive input normalization (see build_policy_input).
        # PPO constructs PolicyNetwork without them and normalizes its inputs
        # itself (PPOAgent._state_to_tensor), so it never calls sample_action /
        # build_policy_input — leaving T=None there is fine.
        self.T = T
        self.finite_horizon = finite_horizon
        self._model = None
        self._optimizer = None
        self._build()

    def _build(self) -> None:
        import torch
        import torch.nn as nn

        layers = []
        in_dim = self.input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, self.n_assets * self.n_actions))
        self._model = nn.Sequential(*layers).float()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """x: (batch, input_dim) → (batch, N, 4) logits."""
        import torch
        out = self._model(x)  # (batch, N*4)
        return out.view(-1, self.n_assets, self.n_actions)

    def sample_action(self, state: State) -> np.ndarray:
        """Sample action from policy network using softmax."""
        import torch
        feat = build_policy_input(state, self.n_assets, self.T, self.finite_horizon)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)  # (1, N, 4)
            probs = torch.softmax(logits, dim=-1)  # (1, N, 4)
            # Sample from categorical
            actions = torch.multinomial(
                probs.squeeze(0),  # (N, 4)
                num_samples=1
            ).squeeze(-1)  # (N,)
        return actions.numpy()


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def build_policy_input(
    state: State,
    n_assets: int,
    T: int | None,
    finite_horizon: bool,
) -> np.ndarray:
    """Flat, scale-normalized input vector for the (neural) PolicyNetwork.

    state.features() = [d, h, ell, r, n_fail] (shape 5N). All columns are ~[0,1]
    except n_fail in [0, T], which would otherwise dominate the input and hurt
    conditioning of the MLP. We rescale the n_fail block by 1/T so every column
    is ~O(1) (matching the static scaling used by the PPO actor/critic).

    When finite_horizon, the normalized epoch t/T is appended (giving 5N+1), so
    the produced vector matches the input_dim the network was built with — this
    also fixes the latent 5N vs 5N+1 mismatch in the actor-critic / NNPolicy
    paths, which previously fed only 5N features.
    """
    denom = max(int(T), 1) if T else 1
    feat = state.features().astype(np.float32)
    feat[4 * n_assets:5 * n_assets] = feat[4 * n_assets:5 * n_assets] / denom
    if finite_horizon:
        feat = np.concatenate([feat, np.array([state.t / denom], dtype=np.float32)])
    return feat


def _asset_features(
    state: State,
    env_config: EnvConfig,
    use_global_context: bool,
    t: int | None = None,
) -> np.ndarray:
    """
    Build per-asset feature matrix.

    Base (6 features per asset):
        [d[i], h[i], ell[i], r[i], n_fail[i], i/(N-1)]

    With global context (9 features per asset):
        + [mean(d), mean(n_fail), t/(T-1)]

    Returns shape (N, F).
    """
    N = env_config.n_assets
    s = state
    idx = np.arange(N) / max(N - 1, 1)  # normalised asset index

    base = np.column_stack([s.d, s.h, s.ell, s.r, s.n_fail, idx])  # (N, 6)

    if use_global_context:
        mean_d = np.full(N, s.d.mean())
        mean_nf = np.full(N, s.n_fail.mean())
        T = env_config.T
        t_val = float(s.t if t is None else t)
        t_norm = np.full(N, t_val / max(T - 1, 1))
        base = np.column_stack([base, mean_d, mean_nf, t_norm])  # (N, 9)

    return base


def _build_dataset(
    states: list[State],
    actions_matrix: np.ndarray,
    env_config: EnvConfig,
    use_global_context: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten (len(states) × N) samples from a list of states and action matrix.
    Returns X: (M, F), y: (M,) int.
    """
    rows = []
    targets = []
    for k, s in enumerate(states):
        feats = _asset_features(s, env_config, use_global_context)  # (N, F)
        rows.append(feats)
        targets.append(actions_matrix[k])  # (N,)
    X = np.vstack(rows)          # (len*N, F)
    y = np.concatenate(targets)  # (len*N,)
    return X.astype(np.float32), y.astype(int)


# ---------------------------------------------------------------------------
# XGBoostPolicy
# ---------------------------------------------------------------------------

class XGBoostPolicy:
    """
    Per-asset XGBClassifier trained over all assets with shared parameters.

    Features per asset:
        [d[i], h[i], ell[i], r[i], n_fail[i], i/(N-1)]
        optionally + [mean(d), mean(n_fail), t/(T-1)]

    Refit from scratch on each fit() call (like XGBoostValueFn).
    """

    DEFAULT_PARAMS: dict = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'n_jobs': -1,
        'verbosity': 0,
    }

    def __init__(
        self,
        n_assets: int,
        env_config: EnvConfig,
        use_global_context: bool = True,
        clf_kwargs: dict | None = None,
    ):
        self.n_assets = n_assets
        self.env_config = env_config
        self.use_global_context = use_global_context
        self._params = {**self.DEFAULT_PARAMS, **(clf_kwargs or {})}
        self._clf = None
        self._fitted = False

    def predict(self, state: State) -> np.ndarray:
        """Return per-asset action array shape (N,) int."""
        if not self._fitted:
            return np.zeros(self.n_assets, dtype=int)
        X = _asset_features(state, self.env_config, self.use_global_context)
        return self._clf.predict(X).astype(int)

    # act() alias so this class can serve as a rollout policy after switching
    def act(self, state: State) -> np.ndarray:
        return self.predict(state)

    def fit(self, states: list[State], actions_matrix: np.ndarray) -> None:
        """
        Refit classifier from scratch.
        actions_matrix: shape (len(states), N) int — rollout-optimal actions.
        """
        import xgboost as xgb

        X, y = _build_dataset(states, actions_matrix, self.env_config, self.use_global_context)
        self._clf = xgb.XGBClassifier(num_class=4, **self._params)
        self._clf.fit(X, y)
        self._fitted = True

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'clf': self._clf,
                'fitted': self._fitted,
                'params': self._params,
            }, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._clf = data['clf']
        self._fitted = data['fitted']
        self._params = data['params']


# ---------------------------------------------------------------------------
# NNPolicy
# ---------------------------------------------------------------------------

class NNPolicy:
    """
    Thin wrapper around PolicyNetwork.

    predict(state) → (N,) int argmax.
    fit(states, actions_matrix): supervised cross-entropy (imitation).
    """

    def __init__(self, network: PolicyNetwork, lr: float = 1e-3):
        """
        network: PolicyNetwork instance.
        lr: learning rate (overrides network.lr for future calls).
        """
        self.network = network
        self.lr = lr

    def predict(self, state: State) -> np.ndarray:
        """Greedy (argmax) action per asset, shape (N,)."""
        import torch
        net = self.network
        feat = build_policy_input(state, net.n_assets, net.T, net.finite_horizon)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.network.forward(x)  # (1, N, 4)
            actions = logits.squeeze(0).argmax(dim=-1)  # (N,)
        return actions.numpy().astype(int)

    # act() alias so this class can serve as a rollout policy after switching
    def act(self, state: State) -> np.ndarray:
        return self.predict(state)

    def fit(self, states: list[State], actions_matrix: np.ndarray) -> None:
        """
        Cross-entropy imitation loss.
        actions_matrix: (B, N) int.
        """
        import torch
        import torch.nn.functional as F

        net = self.network
        X = torch.tensor(
            np.stack([build_policy_input(s, net.n_assets, net.T, net.finite_horizon)
                      for s in states]),
            dtype=torch.float32,
        )  # (B, 5N) or (B, 5N+1)
        Y = torch.tensor(actions_matrix, dtype=torch.long)  # (B, N)

        self.network._model.train()
        logits = self.network.forward(X)  # (B, N, 4)
        B, N, A = logits.shape
        loss = F.cross_entropy(logits.view(B * N, A), Y.view(B * N))

        self.network._optimizer.zero_grad()
        loss.backward()
        self.network._optimizer.step()

    def save(self, path: str) -> None:
        import torch
        if self.network._model is not None:
            torch.save(self.network._model.state_dict(), path)

    def load(self, path: str) -> None:
        import torch
        if self.network._model is not None:
            self.network._model.load_state_dict(torch.load(path))
