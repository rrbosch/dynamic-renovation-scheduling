"""ValueNetwork: simple MLP scalar critic."""
from __future__ import annotations


class ValueNetwork:
    """Simple MLP scalar critic."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self._model = None
        self._build()

    def _build(self) -> None:
        import torch.nn as nn
        layers = []
        in_dim = self.input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self._model = nn.Sequential(*layers).float()

    def parameters(self):
        return self._model.parameters()

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """x: (batch, input_dim) → (batch,)"""
        return self._model(x).squeeze(-1)
