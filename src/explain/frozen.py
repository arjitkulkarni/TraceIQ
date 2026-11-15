"""Explainable forms of the trained models.

GNNExplainer injects edge masks through PyG MessagePassing hooks. The GCN family is trained with
sparse matrix products, which those hooks cannot reach. FrozenGCN runs the same arithmetic through
MessagePassing, using the exact weights the deployed model had at one time step. A test asserts
its outputs equal the deployed model's outputs, so explanations describe the real model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.models.gnn import LEAK, EvolveGCN, StaticGCNNet


class _WeightedPropagate(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add", flow="source_to_target")

    def forward(self, h: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        return self.propagate(edge_index, x=h, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.unsqueeze(-1) * x_j


class FrozenGCN(nn.Module):
    """Two GCN layers with fixed weights. Evaluation mode only: dropout is not applied."""

    def __init__(self, input_layer: nn.Module, weights: list[Tensor], head: nn.Module):
        super().__init__()
        self.input = input_layer
        self.weights = [w.detach() for w in weights]
        self.head = head
        self.layers = nn.ModuleList([_WeightedPropagate() for _ in weights])

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        index, norm = gcn_norm(edge_index, None, x.size(0), improved=False, add_self_loops=True)
        h = F.relu(self.input(x))
        for layer, weight in zip(self.layers, self.weights):
            h = F.leaky_relu(layer(h @ weight, index, norm), LEAK)
        return self.head(h)


@torch.no_grad()
def temporal_weights(model: EvolveGCN, snapshots) -> dict[int, list[Tensor]]:
    """Roll forward from step 1 and record the weights in force at every time step."""
    model.eval()
    state = model.initial_state()
    out: dict[int, list[Tensor]] = {}
    for snapshot in snapshots:
        _, state = model.step(snapshot.x, snapshot.edge_index, state)
        out[snapshot.time_step] = [w.T.clone() for w, _ in state]
    return out


def explainable_model(model: nn.Module, snapshots) -> "callable":
    """Return a function: time_step -> module whose forward is (x, edge_index) -> logits."""
    if isinstance(model, EvolveGCN):
        weights = temporal_weights(model, snapshots)
        return lambda step: FrozenGCN(model.input, weights[step], model.head)
    if isinstance(model, StaticGCNNet):
        frozen = FrozenGCN(model.input, list(model.weights), model.head)
        return lambda step: frozen
    return lambda step: model
