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
            pickle.dump({'model': self._model, 'fitted': self._fitted}, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._model = data['model']
        self._fitted = data['fitted']


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
        X = torch.tensor(self._feats(states), dtype=torch.float32)
        with torch.no_grad():
            return self._model(X).squeeze(-1).numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn.functional as F

        if self._model is None:
            self._build(X.shape[1])

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

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
        }, path)

    def load(self, path: str) -> None:
        import torch
        data = torch.load(path)
        self.hidden_dims = data['hidden_dims']
        self.lr = data['lr']
        if data['state_dict'] is not None:
            self._build(data['input_dim'])
            self._model.load_state_dict(data['state_dict'])
            if data.get('optimizer_state_dict') is not None and self._optimizer is not None:
                self._optimizer.load_state_dict(data['optimizer_state_dict'])
