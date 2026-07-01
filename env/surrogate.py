"""Surrogate TAP model based on XGBoost."""
from __future__ import annotations

import numpy as np
from env.network import NetworkData
from env.tap import TAPSolver


class SurrogateTAP:
    """
    XGBoost surrogate for TAP. Predicts equilibrium flows from binary
    capacity-reduction feature vector.
    """

    def __init__(self, network: NetworkData):
        self.network = network
        self._model = None  # xgboost.XGBRegressor (multioutput via wrapper)
        self._cache: dict[bytes, np.ndarray] = {}  # exact solutions only; predictions excluded

    # ------------------------------------------------------------------
    # TAPSolver protocol
    # ------------------------------------------------------------------

    def solve(self, capacities: np.ndarray) -> np.ndarray:
        """
        Return cached exact solution if available, otherwise predict with XGBoost.
        Predictions are intentionally not cached to avoid contaminating training data.
        """
        key = capacities.tobytes()
        if key in self._cache:
            return self._cache[key]

        if self._model is None:
            raise RuntimeError("SurrogateTAP has no trained model. Call train() first.")

        features = self._capacity_features(capacities)
        pred = self._model.predict(features.reshape(1, -1))
        return pred.flatten()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, tap_solver: TAPSolver, n_samples: int,
              rng: np.random.Generator) -> None:
        """
        Generate random capacity reduction vectors, solve exact TAP,
        train XGBoost on (binary_feature_vector -> flows).
        """
        import xgboost as xgb
        from sklearn.multioutput import MultiOutputRegressor

        net = self.network
        n_edges = net.n_edges
        n_assets = net.n_assets
        asset_idx = net.asset_indices

        X = np.zeros((n_samples, n_assets), dtype=float)
        Y = np.zeros((n_samples, n_edges), dtype=float)

        for i in range(n_samples):
            # Random binary mask: each asset either reduced or not
            mask = rng.integers(0, 2, size=n_assets).astype(bool)
            caps = net.nominal_capacities.copy()
            caps[asset_idx[mask]] *= 0.05  # simulate renovation
            X[i] = mask.astype(float)
            flows = tap_solver.solve(caps)
            Y[i] = flows
            self._cache[caps.tobytes()] = flows.copy()

        base_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            n_jobs=-1,
            verbosity=0,
        )
        self._model = MultiOutputRegressor(base_model, n_jobs=1)
        self._model.fit(X, Y)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load_pretrained(cls, path: str) -> 'SurrogateTAP':
        raise NotImplementedError(
            "Pretrained Amsterdam surrogate not yet available. "
            "Train one first using SurrogateTAP.train()."
        )

    def save(self, path: str) -> None:
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, 'rb') as f:
            self._model = pickle.load(f)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capacity_features(self, capacities: np.ndarray) -> np.ndarray:
        """Binary indicator: is each asset at reduced capacity?"""
        net = self.network
        nominal = net.nominal_capacities[net.asset_indices]
        reduced = capacities[net.asset_indices]
        return (reduced < nominal * 0.9).astype(float)
