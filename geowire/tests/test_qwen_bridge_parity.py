from __future__ import annotations

import torch
from torch import nn

from geowire.geometry.graph_builder import self_loop_graph
from geowire.models.geowire import GeoWireTransport
from geowire.models.qwen3vl_bridge import Qwen3VLGeoWireForConditionalGeneration


class _DummyOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _DummyInner(nn.Module):
    def __init__(self, hidden: torch.Tensor) -> None:
        super().__init__()
        self.visual_hidden = nn.Parameter(hidden.clone())

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return (self.visual_hidden,), ()

    def forward(self, *, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, labels=None, **kwargs):
        image_embeds, _ = self.get_image_features(pixel_values, image_grid_thw)
        return _DummyOutput(torch.cat(image_embeds, dim=0))


class _DummyBase(nn.Module):
    def __init__(self, hidden: torch.Tensor) -> None:
        super().__init__()
        self.model = _DummyInner(hidden)
        self.config = object()

    def get_input_embeddings(self):
        return nn.Embedding(4, 4)

    def forward(self, **kwargs):
        return self.model(**kwargs)


def test_qwen_bridge_alpha_zero_parity() -> None:
    hidden = torch.randn(6, 8)
    base = _DummyBase(hidden)
    geowire = GeoWireTransport(hidden_size=8, num_blocks=2)
    bridge = Qwen3VLGeoWireForConditionalGeneration(base, geowire, token_layout_builder=None)
    graph = self_loop_graph(num_nodes=6)

    before_method = base.model.get_image_features
    base_out = base(pixel_values=torch.empty(1), image_grid_thw=torch.tensor([[1, 2, 3]])).logits
    bridge_out = bridge(
        graph=graph,
        pixel_values=torch.empty(1),
        image_grid_thw=torch.tensor([[1, 2, 3]]),
    ).logits

    assert torch.equal(base_out, bridge_out)
    assert base.model.get_image_features == before_method
