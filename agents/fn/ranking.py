"""Ranking-based value function using XGBoost LambdaRank."""
from __future__ import annotations

import numpy as np
import pickle

from agents.fn.value_fn import ValueFn
from env.mdp import State


class RankingValueFn(ValueFn):
    """
    XGBoost LambdaRank value function.
    Lower scores = better cost-to-go (lower is better convention).
    Drop-in replacement for XGBoostValueFn in DQNAgent.
    """

    prefers_full_dataset: bool = True

    DEFAULT_PARAMS = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'objective': 'rank:pairwise',
        'n_jobs': -1,
        'verbosity': 0,
    }

    def __init__(self, xgb_params: dict | None = None, finite_horizon: bool = True):
        self._params = {**self.DEFAULT_PARAMS, **(xgb_params or {})}
        import xgboost as xgb
        self._model = xgb.XGBRanker(**self._params)
        self._fitted = False
        self.finite_horizon = finite_horizon
        self.last_rank_errors: np.ndarray | None = None

    def predict(self, states: list[State]) -> np.ndarray:
        """Returns ranking scores (lower = better cost-to-go)."""
        if not self._fitted:
            return np.zeros(len(states))
        X = self._feats(states)
        return self._model.predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit with pairwise ranking objective.
        Constructs a single query group from all samples.
        y values are treated as relevance labels (higher = worse, so we negate).
        """
        import xgboost as xgb

        # For LambdaRank: higher relevance = preferred. Since lower cost-to-go = better,
        # we negate y so that samples with lower y get higher relevance.
        y_rel = -y

        # Normalize to non-negative integers for ranking
        y_min = y_rel.min()
        y_rel_shifted = y_rel - y_min

        # Single group: all samples in one query
        group = np.array([len(X)], dtype=int)

        self._model = xgb.XGBRanker(**self._params)
        self._model.fit(X, y_rel_shifted, group=group)
        self._fitted = True

        # Compute rank displacement as a meaningful pred_error proxy in [0, 1].
        pred = self._model.predict(X)
        n = len(y)
        true_ranks = np.argsort(np.argsort(y))
        pred_ranks = np.argsort(np.argsort(pred))
        self.last_rank_errors = np.abs(true_ranks - pred_ranks) / n

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({'model': self._model, 'fitted': self._fitted}, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._model = data['model']
        self._fitted = data['fitted']
