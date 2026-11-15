"""Neural models with a shared shape: input projection, two graph layers, binary head.

mlp          No edges. Ablation: removes the graph.
graphsage    Static neighborhood aggregation. No state between snapshots.
static_gcn   GCN with fixed weights. Ablation: EvolveGCN with evolution removed.
evolvegcn_o  Re-implementation of EvolveGCN-O (Pareja et al. 2020): an LSTM evolves GCN weights.
evolvegcn_h  Re-implementation of EvolveGCN-H: a GRU evolves GCN weights from node summaries.

Temporal state is the GCN weight matrix itself. It carries no node identity, so nothing here
follows a transaction, wallet or account from one snapshot to the next.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

TEMPORAL_MODELS = ("evolvegcn_o", "evolvegcn_h")
GRAPH_MODELS = ("graphsage", "static_gcn", *TEMPORAL_MODELS)
ALL_MODELS = ("mlp", *GRAPH_MODELS)
LEAK = 0.2  # keeps units alive during long rollouts; plain ReLU collapsed to a constant output


def normalized_adjacency(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Sparse GCN operator. Messages follow edge_index from source to destination."""
    index, weight = gcn_norm(edge_index, None, num_nodes, improved=False, add_self_loops=True)
    return torch.sparse_coo_tensor(
        index.flip(0), weight, (num_nodes, num_nodes), check_invariants=False
    ).coalesce()


class Head(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h).squeeze(-1)


class MLPNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)])
        self.dropout = dropout
        self.head = Head(hidden)

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = F.relu(self.input(x))
        for layer in self.layers:
            h = F.relu(layer(F.dropout(h, self.dropout, self.training)))
        return h

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.head(self.embed(x, edge_index))


class GraphSAGENet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([SAGEConv(hidden, hidden), SAGEConv(hidden, hidden)])
        self.dropout = dropout
        self.head = Head(hidden)

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = F.relu(self.input(x))
        for layer in self.layers:
            h = F.relu(layer(F.dropout(h, self.dropout, self.training), edge_index))
        return h

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.head(self.embed(x, edge_index))


class StaticGCNNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(hidden, hidden)) for _ in range(2)])
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        self.dropout = dropout
        self.head = Head(hidden)

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        adjacency = normalized_adjacency(edge_index, x.size(0))
        h = F.relu(self.input(x))
        for weight in self.weights:
            h = F.dropout(h, self.dropout, self.training)
            h = F.leaky_relu(torch.sparse.mm(adjacency, h @ weight), LEAK)
        return h

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.head(self.embed(x, edge_index))


class EvolveGCN(nn.Module):
    """Weights W_t per layer. Rows are input channels and columns are output channels.

    O variant: W_t = LSTM(W_{t-1}), with each column treated as one sequence element.
    H variant: W_t = GRU(summary of H_t, W_{t-1}), with top-k node summaries.
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float, variant: str):
        super().__init__()
        if variant not in ("o", "h"):
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.hidden = hidden
        self.dropout = dropout
        self.input = nn.Linear(in_dim, hidden)
        self.initial = nn.ParameterList([nn.Parameter(torch.empty(hidden, hidden)) for _ in range(2)])
        for weight in self.initial:
            nn.init.xavier_uniform_(weight)
        if variant == "o":
            self.cells = nn.ModuleList([nn.LSTMCell(hidden, hidden) for _ in range(2)])
        else:
            self.cells = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(2)])
            self.scorers = nn.ParameterList([nn.Parameter(torch.empty(hidden)) for _ in range(2)])
            for scorer in self.scorers:
                nn.init.normal_(scorer, std=1.0 / hidden**0.5)
        self.head = Head(hidden)

    def initial_state(self) -> list[tuple[Tensor, Tensor | None]]:
        """Reset point: every new sequence starts from the learned W_0."""
        if self.variant == "o":
            return [(w.T, torch.zeros_like(w.T)) for w in self.initial]
        return [(w.T, None) for w in self.initial]

    def _summary(self, h: Tensor, layer: int) -> Tensor:
        """Top-k pooling: one hidden-sized vector per weight column."""
        scorer = self.scorers[layer]
        scores = (h @ scorer) / scorer.norm()
        k = min(self.hidden, h.size(0))
        top = torch.topk(scores, k).indices
        summary = h[top] * torch.tanh(scores[top]).unsqueeze(-1)
        if k < self.hidden:
            summary = F.pad(summary, (0, 0, 0, self.hidden - k))
        return summary

    def step(self, x: Tensor, edge_index: Tensor, state: list) -> tuple[Tensor, list]:
        """Advance one snapshot. Labels are never used, so state can roll through any step."""
        adjacency = normalized_adjacency(edge_index, x.size(0))
        h = F.relu(self.input(x))
        new_state = []
        for i, cell in enumerate(self.cells):
            weight_t, memory = state[i]
            if self.variant == "o":
                weight_t, memory = cell(weight_t, (weight_t, memory))
            else:
                weight_t = cell(self._summary(h, i), weight_t)
            new_state.append((weight_t, memory))
            h = F.dropout(h, self.dropout, self.training)
            h = F.leaky_relu(torch.sparse.mm(adjacency, h @ weight_t.T), LEAK)
        return h, new_state

    @staticmethod
    def detach_state(state: list) -> list:
        return [(w.detach(), None if m is None else m.detach()) for w, m in state]


def build_model(name: str, in_dim: int, hidden: int, dropout: float) -> nn.Module:
    if name == "mlp":
        return MLPNet(in_dim, hidden, dropout)
    if name == "graphsage":
        return GraphSAGENet(in_dim, hidden, dropout)
    if name == "static_gcn":
        return StaticGCNNet(in_dim, hidden, dropout)
    if name == "evolvegcn_o":
        return EvolveGCN(in_dim, hidden, dropout, "o")
    if name == "evolvegcn_h":
        return EvolveGCN(in_dim, hidden, dropout, "h")
    raise ValueError(f"unknown model: {name}")
