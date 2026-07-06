from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as torch_dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(device_arg: str) -> DistributedContext:
    """Initialize torchrun-style distributed training from environment variables."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        if enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(device_arg)
    else:
        device = torch.device(device_arg)
    if enabled and not torch_dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        torch_dist.init_process_group(backend=backend)
    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled:
        torch_dist.barrier()


def cleanup(ctx: DistributedContext) -> None:
    if ctx.enabled and torch_dist.is_initialized():
        torch_dist.destroy_process_group()


@torch.no_grad()
def broadcast_parameters(module: torch.nn.Module, ctx: DistributedContext, *, only_trainable: bool = False) -> None:
    if not ctx.enabled:
        return
    for parameter in module.parameters():
        if only_trainable and not parameter.requires_grad:
            continue
        torch_dist.broadcast(parameter.data, src=0)


@torch.no_grad()
def average_gradients(module: torch.nn.Module, ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        torch_dist.all_reduce(parameter.grad, op=torch_dist.ReduceOp.SUM)
        parameter.grad.div_(ctx.world_size)
