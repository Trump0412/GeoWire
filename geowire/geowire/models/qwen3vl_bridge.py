from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import torch
from torch import nn

from geowire.types import SparseGraph


@dataclass(frozen=True)
class QwenInspection:
    model_class: str
    candidate_modules: dict[str, str]


def inspect_qwen_modules(model: nn.Module) -> QwenInspection:
    candidates: dict[str, str] = {}
    keywords = ("visual", "vision", "merger", "projector", "language", "model")
    for name, module in model.named_modules():
        if any(k in name.lower() for k in keywords):
            try:
                sig = str(inspect.signature(module.forward))
            except (TypeError, ValueError):
                sig = "<unavailable>"
            candidates[name] = f"{module.__class__.__name__}{sig}"
    return QwenInspection(model.__class__.__name__, candidates)


def graph_to_device(graph: SparseGraph, device: torch.device) -> SparseGraph:
    return SparseGraph(
        num_nodes=graph.num_nodes,
        dst=graph.dst.to(device),
        src=graph.src.to(device),
        weight=graph.weight.to(device),
        edge_type=graph.edge_type.to(device),
        reproj_error=graph.reproj_error.to(device),
        cycle_error=graph.cycle_error.to(device),
        visibility=graph.visibility.to(device),
        confidence=graph.confidence.to(device),
    )


class Qwen3VLGeoWireForConditionalGeneration(nn.Module):
    """Qwen3-VL bridge that inserts GeoWire into the image feature path.

    The installed Transformers implementation owns token replacement, RoPE,
    deepstack features, language layers and generation. This bridge wraps only
    `base_model.model.get_image_features`: split image features are concatenated,
    passed through GeoWire once, and split back before the original forward
    continues. With all GeoWire alphas at zero, the wrapped model should match
    the base model exactly.
    """

    def __init__(self, base_model: nn.Module, geowire: nn.Module, token_layout_builder) -> None:
        super().__init__()
        self.base_model = base_model
        self.geowire = geowire
        self.token_layout_builder = token_layout_builder
        if not hasattr(base_model, "model") or not hasattr(base_model.model, "get_image_features"):
            raise TypeError("base_model must expose model.get_image_features")

    @property
    def config(self):
        return self.base_model.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def forward(
        self,
        *args,
        graph: SparseGraph,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ):
        original = self.base_model.model.get_image_features

        def patched_get_image_features(_model_self, pixel_values_arg, image_grid_thw_arg=None):
            image_embeds, deepstack_image_embeds = original(pixel_values_arg, image_grid_thw_arg)
            split_sizes = [int(x.shape[0]) for x in image_embeds]
            concat = torch.cat(image_embeds, dim=0)
            if graph.num_nodes != concat.shape[0]:
                raise ValueError(f"GeoWire graph has {graph.num_nodes} nodes but Qwen produced {concat.shape[0]} image tokens")
            geowire_dtype = next(self.geowire.parameters()).dtype
            transported = self.geowire(concat.to(dtype=geowire_dtype), graph_to_device(graph, concat.device))
            transported = transported.to(dtype=concat.dtype)
            return tuple(torch.split(transported, split_sizes, dim=0)), deepstack_image_embeds

        self.base_model.model.get_image_features = MethodType(patched_get_image_features, self.base_model.model)
        try:
            return self.base_model(
                *args,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                **kwargs,
            )
        finally:
            self.base_model.model.get_image_features = original

    def generate(self, *args, graph: SparseGraph, **kwargs):
        original_forward: Callable[..., Any] = self.forward

        def patched_forward(*forward_args, **forward_kwargs):
            return original_forward(*forward_args, graph=graph, **forward_kwargs)

        old_forward = self.base_model.forward
        self.base_model.forward = MethodType(lambda _self, *a, **kw: patched_forward(*a, **kw), self.base_model)
        try:
            return self.base_model.generate(*args, **kwargs)
        finally:
            self.base_model.forward = old_forward
