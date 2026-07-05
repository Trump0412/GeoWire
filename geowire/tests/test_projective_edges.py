from __future__ import annotations

import torch

from geowire.geometry.projective_edges import bilinear_depth_sample, project_world_points


def test_identity_projection() -> None:
    world = torch.tensor([[0.0, 0.0, 2.0], [1.0, 1.0, 2.0]])
    extrinsic = torch.eye(4)
    intrinsic = torch.tensor([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]])
    xy, valid = project_world_points(world, extrinsic, intrinsic)
    assert valid.tolist() == [True, True]
    assert torch.allclose(xy[0], torch.tensor([10.0, 20.0]))
    assert torch.allclose(xy[1], torch.tensor([11.0, 21.0]))


def test_reject_negative_z() -> None:
    world = torch.tensor([[0.0, 0.0, -1.0]])
    xy, valid = project_world_points(world, torch.eye(4), torch.eye(3))
    assert xy.shape == (1, 2)
    assert valid.tolist() == [False]


def test_bilinear_depth_sample() -> None:
    depth = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    sample, valid = bilinear_depth_sample(depth, torch.tensor([[1.0, 1.0], [3.0, 1.0]]))
    assert valid.tolist() == [True, False]
    assert torch.allclose(sample[0], torch.tensor(4.0))
