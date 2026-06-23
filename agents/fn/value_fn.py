"""Value function approximators."""
from __future__ import annotations

import numpy as np
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from env.mdp import State

if TYPE_CHECKING:
    pass


class ValueFn(ABC):
    prefers_full_dataset: bool = False  # if True, trainer passes full buffer instead of batch_size sample
    finite_horizon: bool = True   # set by subclass __init__

    # --- Per-epoch baseline (advantage target) -------------------------------
    # When enabled, the value function learns the *advantage* over a per-epoch
    # baseline b(t) = mean cost-to-go at epoch t, instead of the raw cost-to-go.
    # The dominant variance in the ADP target (mc_return - cost) is the time
    # trend (early epochs have huge remaining cost-to-go, late epochs tiny);
    # subtracting b(t) removes it so the residual the model fits is closer to the
    # act-vs-wait signal the action search must resolve. This is purely a target
    # reparameterisation: `fit_with_baseline` subtracts b(t) before fitting and
    # `predict` adds it back, so `predict` always returns a full cost-to-go
    # estimate. Q(s,a) = C(s,a) + V'(s_post) in the action generators stays a
    # valid cost estimator unchanged, for BOTH XGBoost and the neural VF.
    # b(t) is held here on the base class so every subclass inherits it.
    _baseline: np.ndarray | None = None   # shape (max_t+1,) lookup table over epoch
    _baseline_enabled: bool = False

    def set_baseline_enabled(self, enabled: bool) -> None:
        self._baseline_enabled = bool(enabled)

    def _fit_baseline(self, t_idx: np.ndarray, y: np.ndarray) -> None:
        """Estimate b(t) = mean(y | epoch=t) from the current training data.

        Frozen on the FIRST call (like the NN standardization stats) so the
        baseline space stays consistent across the many small fit() calls in the
        training loop. Buckets with no samples fall back to the global mean."""
        if self._baseline is not None:
            return
        t_idx = np.asarray(t_idx, dtype=np.int64)
        t_idx = np.maximum(t_idx, 0)
        max_t = int(t_idx.max())
        gmean = float(y.mean())
        b = np.full(max_t + 1, gmean, dtype=np.float64)
        sums = np.bincount(t_idx, weights=y, minlength=max_t + 1)
        cnts = np.bincount(t_idx, minlength=max_t + 1)
        nz = cnts > 0
        b[nz] = sums[nz] / cnts[nz]
        self._baseline = b

    def _baseline_for(self, states: list[State]) -> np.ndarray:
        """Look up b(t) for each state's epoch, shape (len(states),)."""
        if self._baseline is None:
            return np.zeros(len(states))
        t = np.fromiter((s.t for s in states), dtype=np.int64, count=len(states))
        t = np.clip(t, 0, len(self._baseline) - 1)
        return self._baseline[t]

    def fit_targets(self, states: list[State], y: np.ndarray) -> None:
        """Fit on (post-decision states, cost-to-go targets y).

        Wraps the subclass `fit(X, y)`. When the per-epoch baseline is enabled,
        b(t) is frozen on the first call and the model is fit on the advantage
        y - b(t); `predict` adds b(t) back. When disabled, this is exactly the
        old behaviour: fit on the raw cost-to-go target."""
        X = self._feats(states)
        if self._baseline_enabled:
            t_idx = np.fromiter((s.t for s in states), dtype=np.int64,
                                count=len(states))
            self._fit_baseline(t_idx, y)
            y = y - self._baseline_for(states)
        self.fit(X, y)

    def _feats(self, states: list[State]) -> np.ndarray:
        """Extract feature matrix, optionally appending t."""
        X = np.concatenate([
            np.stack([s.d      for s in states]),
            np.stack([s.h      for s in states]),
            np.stack([s.ell    for s in states]),
            np.stack([s.r      for s in states]),
            np.stack([s.n_fail for s in states]),
        ], axis=1)
        if self.finite_horizon:
            t_col = np.fromiter((s.t for s in states), dtype=np.float64,
                                count=len(states)).reshape(-1, 1)
            X = np.concatenate([X, t_col], axis=1)
        return X

    @abstractmethod
    def predict(self, states: list[State]) -> np.ndarray:
        """Returns predicted values, shape (len(states),)."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the value function. X: (n, 5N+1) or (n, 5N), y: (n,)."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


# ---------------------------------------------------------------------------
# XGBoost value function
# ---------------------------------------------------------------------------

class XGBoostValueFn(ValueFn):
    """Wraps xgboost.XGBRegressor. Refit from scratch on each fit() call."""

    prefers_full_dataset: bool = True  # retrains from scratch — needs all buffered data

    DEFAULT_PARAMS = {
        'n_estimators': 10000,
        'max_depth': 12,
        'learning_rate': 0.2,
        'subsample': 0.8,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 10,
    }

    def __init__(self, xgb_params: dict | None = None, finite_horizon: bool = True):
        params = {**self.DEFAULT_PARAMS, **(xgb_params or {})}
        import xgboost as xgb
        self._model = xgb.XGBRegressor(**params)
        self._params = params
        self._fitted = False
        self.prmse = 0
        self.rank_error = 0.0
        self.finite_horizon = finite_horizon

    def predict(self, states: list[State]) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(states))
        X = self._feats(states)
        preds = self._model.predict(X)
        if self._baseline_enabled:
            # Model learned the advantage y - b(t); add b(t) back so predict
            # returns a full cost-to-go estimate (Q reconstruction unchanged).
            preds = preds + self._baseline_for(states)
        preds[preds < 0.01] = 0.01
        return preds

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator = None) -> None:
        import xgboost as xgb
        if rng is None:
            rng = np.random.default_rng(42)
        self._model = xgb.XGBRegressor(**self._params)
        # 10% validation split for early stopping
        n_val = max(1, int(0.1 * len(X)))
        idx = rng.permutation(len(X))
        X_train, y_train = X[idx[n_val:]], y[idx[n_val:]]
        X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]
        self._model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self._fitted = True
        self.prmse = self._model.best_score / y.mean()
        y_pred = self._model.predict(X)
        n = len(y)
        true_ranks = np.argsort(np.argsort(y))
        pred_ranks = np.argsort(np.argsort(y_pred))
        self.rank_error = float(np.mean(np.abs(true_ranks - pred_ranks) / n))
        print(f"proportional RMSE: {self.prmse:.4f}  |  rank error: {self.rank_error:.4f}")

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'model': self._model, 'fitted': self._fitted,
                         'baseline': self._baseline,
                         'baseline_enabled': self._baseline_enabled}, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._model = data['model']
        self._fitted = data['fitted']
        self._baseline = data.get('baseline')
        self._baseline_enabled = data.get('baseline_enabled', False)


# ---------------------------------------------------------------------------
# Neural value function (stub)
# ---------------------------------------------------------------------------

class NeuralValueFn(ValueFn):
    """MLP in torch. Input dim = 5N (or 5N+1 with finite_horizon=True), output dim = 1."""

    def __init__(self, hidden_dims: list[int] = (256, 256), lr: float = 1e-3,
                 finite_horizon: bool = True):
        self.hidden_dims = list(hidden_dims)
        self.lr = lr
        self._model = None
        self._optimizer = None
        self._input_dim: int | None = None
        self.finite_horizon = finite_horizon
        # Standardization stats. The targets here are euro cost-to-go values of
        # magnitude ~1e9-1e11 and the input columns span very different scales
        # (d in [0,1], n_fail / t up to T). A plain-MSE MLP with grad clipping
        # cannot fit raw targets of that magnitude (loss ~1e22, clipped grads
        # never move the net -> predictions collapse to a near-constant, which
        # makes the action generator unable to rank candidates -> do-nothing
        # policy). Tree models (XGBoost) are scale-invariant and so do not need
        # this. We z-score inputs and targets at fit time and invert on predict.
        # Stats are frozen on the FIRST fit() call so the persistent network
        # (reused across the many small fit() calls in the training loop) always
        # sees a consistent input scaling and target space.
        self._x_mean = None
        self._x_std = None
        self._y_mean: float | None = None
        self._y_std: float | None = None
        self._stats_frozen: bool = False

    def _freeze_stats(self, X: np.ndarray, y: np.ndarray) -> None:
        self._x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)
        x_std[x_std < 1e-8] = 1.0  # guard constant columns (e.g. h all-zero early)
        self._x_std = x_std
        self._y_mean = float(y.mean())
        y_std = float(y.std())
        self._y_std = y_std if y_std > 1e-8 else 1.0
        self._stats_frozen = True

    def _build(self, input_dim: int) -> None:
        import torch
        import torch.nn as nn

        layers = []
        in_dim = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self._model = nn.Sequential(*layers).float()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._input_dim = input_dim

    def predict(self, states: list[State]) -> np.ndarray:
        import torch
        if self._model is None:
            return np.zeros(len(states))
        X = self._feats(states)
        if self._stats_frozen:
            X = (X - self._x_mean) / self._x_std
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = self._model(X_t).squeeze(-1).numpy()
        if self._stats_frozen:
            preds = preds * self._y_std + self._y_mean
        if self._baseline_enabled:
            # Model learned the advantage y - b(t); add b(t) back so predict
            # returns a full cost-to-go estimate (Q reconstruction unchanged).
            preds = preds + self._baseline_for(states)
        # Cost-to-go is non-negative; clamp like XGBoostValueFn to avoid the
        # action generator being misled by spurious negative values.
        preds = np.maximum(preds, 0.01)
        return preds

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn.functional as F

        if self._model is None:
            self._build(X.shape[1])

        # Freeze input/target standardization stats on the first fit, then reuse
        # them for the lifetime of this (persistent) network. See __init__ note.
        if not self._stats_frozen:
            self._freeze_stats(X, y)

        Xn = (X - self._x_mean) / self._x_std
        yn = (y - self._y_mean) / self._y_std

        X_t = torch.tensor(Xn, dtype=torch.float32)
        y_t = torch.tensor(yn, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        self._model.train()
        for epoch in range(50):
            for xb, yb in loader:
                self._optimizer.zero_grad()
                pred = self._model(xb).squeeze(-1)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()

    def save(self, path: str) -> None:
        import torch
        torch.save({
            'state_dict': self._model.state_dict() if self._model else None,
            'optimizer_state_dict': self._optimizer.state_dict() if self._optimizer else None,
            'hidden_dims': self.hidden_dims,
            'lr': self.lr,
            'input_dim': self._input_dim,
            'stats_frozen': self._stats_frozen,
            'x_mean': self._x_mean,
            'x_std': self._x_std,
            'y_mean': self._y_mean,
            'y_std': self._y_std,
            'baseline': self._baseline,
            'baseline_enabled': self._baseline_enabled,
        }, path)

    def load(self, path: str) -> None:
        import torch
        # weights_only=False: checkpoint stores numpy standardization stats
        # (x_mean/x_std/...) alongside tensors. These are locally produced,
        # trusted files; torch>=2.6 defaults weights_only=True which rejects them.
        data = torch.load(path, weights_only=False)
        self.hidden_dims = data['hidden_dims']
        self.lr = data['lr']
        self._stats_frozen = data.get('stats_frozen', False)
        self._x_mean = data.get('x_mean')
        self._x_std = data.get('x_std')
        self._y_mean = data.get('y_mean')
        self._y_std = data.get('y_std')
        self._baseline = data.get('baseline')
        self._baseline_enabled = data.get('baseline_enabled', False)
        if data['state_dict'] is not None:
            self._build(data['input_dim'])
            self._model.load_state_dict(data['state_dict'])
            if data.get('optimizer_state_dict') is not None and self._optimizer is not None:
                self._optimizer.load_state_dict(data['optimizer_state_dict'])
