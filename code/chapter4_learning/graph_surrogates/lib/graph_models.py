"""Differentiable surrogate architectures for Chapter 4.

All models consume the same 13 standardized `DummyParams` values and emit one
logit. The graph-like models use fixed clevis topology and torch arithmetic only,
so autograd can still compute gradients with respect to the original parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data_utils import FEATURE_LIST

N_FEATURES = len(FEATURE_LIST)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[std < 1e-9] = 1.0
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class MLP64Control(nn.Module):
    architecture_id = "mlp64_control"

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WideMLPControl(nn.Module):
    architecture_id = "wide_mlp_control"

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _FixedGraphMPNN(nn.Module):
    """Small fixed-topology message-passing network."""

    def __init__(
        self,
        *,
        architecture_id: str,
        node_groups: list[list[int]],
        edge_pairs: list[tuple[int, int]],
        hidden_dim: int = 48,
        rounds: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.architecture_id = architecture_id
        self.node_groups = node_groups
        self.edge_pairs = edge_pairs
        self.rounds = rounds
        self.n_nodes = len(node_groups)

        max_group = max(len(g) for g in node_groups)
        self.max_group = max_group
        self.node_encoder = nn.Sequential(
            nn.Linear(max_group + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + N_FEATURES, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _node_tensor(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        nodes = []
        for group in self.node_groups:
            vals = x[:, group]
            if len(group) < self.max_group:
                pad = torch.zeros(
                    batch,
                    self.max_group - len(group),
                    device=x.device,
                    dtype=x.dtype,
                )
                vals = torch.cat([vals, pad], dim=1)
            summary = torch.stack(
                [vals.mean(dim=1), vals.std(dim=1, unbiased=False), vals.amax(dim=1)],
                dim=1,
            )
            nodes.append(torch.cat([vals, summary], dim=1))
        return torch.stack(nodes, dim=1)

    def _edge_features(self, h0: torch.Tensor, src: int, dst: int) -> torch.Tensor:
        a = h0[:, src, :]
        b = h0[:, dst, :]
        return torch.stack(
            [
                a.mean(dim=1),
                b.mean(dim=1),
                (a - b).mean(dim=1),
                (a * b).mean(dim=1),
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        node_raw = self._node_tensor(x)
        h = self.node_encoder(node_raw)
        h0 = node_raw
        batch = x.shape[0]

        for _ in range(self.rounds):
            agg = torch.zeros_like(h)
            for src, dst in self.edge_pairs:
                edge = self.edge_encoder(self._edge_features(h0, src, dst))
                msg = self.message(torch.cat([h[:, src, :], h[:, dst, :], edge], dim=1))
                agg[:, dst, :] = agg[:, dst, :] + msg
            h = self.update(agg.reshape(batch * self.n_nodes, -1), h.reshape(batch * self.n_nodes, -1))
            h = h.reshape(batch, self.n_nodes, -1)

        pooled_mean = h.mean(dim=1)
        pooled_max = h.amax(dim=1)
        return self.head(torch.cat([pooled_mean, pooled_max, x], dim=1)).squeeze(-1)


class PartGraphMPNN(_FixedGraphMPNN):
    architecture_id = "part_graph_mpnn_v1"

    def __init__(self, dropout: float = 0.1):
        super().__init__(
            architecture_id=self.architecture_id,
            node_groups=[
                [0, 1, 2, 3, 4, 5, 12],  # bracket / roof / main hole
                [6, 7, 8, 9],             # main pin / cross-hole
                [10, 11],                 # splint
            ],
            edge_pairs=[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)],
            hidden_dim=48,
            rounds=2,
            dropout=dropout,
        )


class ConstraintGraphMPNN(_FixedGraphMPNN):
    architecture_id = "constraint_graph_mpnn_v1"

    def __init__(self, dropout: float = 0.1):
        super().__init__(
            architecture_id=self.architecture_id,
            node_groups=[
                [0, 1, 2, 3],      # frame
                [0, 12],           # roof
                [4, 5],            # main hole
                [6, 7],            # main pin
                [8, 9],            # cross-hole
                [10, 11],          # splint shaft
                [10, 11, 12],      # splint head / roof interaction
            ],
            edge_pairs=[
                (0, 1), (1, 0),
                (0, 2), (2, 0),
                (2, 3), (3, 2),
                (3, 4), (4, 3),
                (4, 5), (5, 4),
                (5, 6), (6, 5),
                (1, 6), (6, 1),
                (3, 6), (6, 3),
            ],
            hidden_dim=56,
            rounds=3,
            dropout=dropout,
        )


class EdgePoolGraph(nn.Module):
    architecture_id = "edge_pool_graph_v1"

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # Each edge contains a compact differentiable relation between params.
        self.edge_index_groups = [
            (7, 5),    # main pin radius vs hole radius
            (10, 8),   # splint radius vs cross-hole radius
            (6, 1),    # main pin length vs outer span
            (11, 7),   # splint length vs main pin radius
            (12, 1),   # overhang vs bracket span
            (4, 2),    # hole offset vs leg length
            (9, 6),    # cross-hole offset vs pin length
            (10, 5),   # splint head vs main-hole bore proxy
        ]
        edge_dim = 5
        hidden = 64
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + N_FEATURES, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _edge_tensor(self, x: torch.Tensor) -> torch.Tensor:
        edges = []
        for a_idx, b_idx in self.edge_index_groups:
            a = x[:, a_idx]
            b = x[:, b_idx]
            edges.append(torch.stack([a, b, a - b, a + b, a * b], dim=1))
        return torch.stack(edges, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.edge_tensor = self._edge_tensor(x)
        h = self.edge_mlp(e)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1), x], dim=1)
        return self.head(pooled).squeeze(-1)


MODEL_REGISTRY = {
    "mlp64_control": MLP64Control,
    "wide_mlp_control": WideMLPControl,
    "part_graph_mpnn_v1": PartGraphMPNN,
    "constraint_graph_mpnn_v1": ConstraintGraphMPNN,
    "edge_pool_graph_v1": EdgePoolGraph,
}


def create_model(architecture: str, dropout: float = 0.1) -> nn.Module:
    try:
        cls = MODEL_REGISTRY[architecture]
    except KeyError as exc:
        raise ValueError(f"Unknown Chapter 4 architecture: {architecture}") from exc
    return cls(dropout=dropout)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
