from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from geowire.geometry.transforms import make_frame_transform
from geowire.types import FrameTransform, VGGTGeometry

class VGGTProvider:
    """Thin wrapper around pinned VGGT source.

    VGGT execution is isolated here so the main GeoWire code does not edit upstream
    files or use VGGT hidden states as semantic values.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        source_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | str = torch.bfloat16,
        image_size: int = 518,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.source_path = Path(source_path) if source_path is not None else None
        self.device = torch.device(device)
        self.dtype = _dtype_from_name(dtype)
        self.image_size = int(image_size)
        self._model: torch.nn.Module | None = None

    def assert_available(self) -> None:
        if self.source_path is not None and not self.source_path.exists():
            raise FileNotFoundError(f"VGGT source path is not available: {self.source_path}")
        path = Path(self.checkpoint)
        if path.exists():
            return
        if "/" in self.checkpoint:
            return
        raise FileNotFoundError(f"VGGT checkpoint is not available: {self.checkpoint}")

    def load_model(self) -> torch.nn.Module:
        self.assert_available()
        if self._model is not None:
            return self._model
        if self.source_path is not None:
            sys.path.insert(0, str(self.source_path))
        from vggt.models.vggt import VGGT

        checkpoint_path = Path(self.checkpoint)
        if checkpoint_path.exists():
            model = VGGT()
            weights_path = _resolve_local_weight(checkpoint_path)
            if weights_path.suffix == ".safetensors":
                state = load_file(str(weights_path))
            else:
                state = torch.load(weights_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            model.load_state_dict(state, strict=False)
        else:
            model = VGGT.from_pretrained(self.checkpoint)
        model.eval().to(self.device)
        self._model = model
        return model

    @torch.inference_mode()
    def infer_geometry(
        self,
        frame_paths: list[str],
        *,
        frame_transforms: tuple[FrameTransform, ...] | None = None,
    ) -> VGGTGeometry:
        """Run canonical VGGT camera/depth/point inference for one ordered clip."""

        model = self.load_model()
        if self.source_path is not None:
            sys.path.insert(0, str(self.source_path))
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = load_and_preprocess_images(frame_paths).to(self.device)
        if frame_transforms is None:
            frame_transforms = _default_frame_transforms(frame_paths, self.image_size)
        with _autocast(self.device, self.dtype):
            predictions = model(images)
        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth = predictions["depth"].squeeze(0).squeeze(-1).float()
        depth_conf = predictions["depth_conf"].squeeze(0).float()
        world_points_head = predictions["world_points"].squeeze(0).float()
        point_conf = predictions["world_points_conf"].squeeze(0).float()
        world_points_unproj = unproject_depth_map_to_point_map(
            depth,
            extrinsic.squeeze(0).float(),
            intrinsic.squeeze(0).float(),
        )
        return VGGTGeometry(
            extrinsic_cw=extrinsic.squeeze(0).detach().cpu().float(),
            intrinsic=intrinsic.squeeze(0).detach().cpu().float(),
            depth=depth.detach().cpu(),
            depth_conf=depth_conf.detach().cpu(),
            world_points_head=world_points_head.detach().cpu(),
            world_points_unproj=world_points_unproj.detach().cpu().float(),
            point_conf=point_conf.detach().cpu(),
            frame_transforms=frame_transforms,
            track_xy=None,
            track_vis=None,
            track_conf=None,
            track_anchor_frames=None,
            track_query_token_ids=None,
        )

    @torch.inference_mode()
    def track_from_anchor(
        self,
        frame_paths: list[str],
        anchor_frame: int,
        query_xy_vggt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Track points from an arbitrary anchor frame via full sequence permutation."""

        model = self.load_model()
        if self.source_path is not None:
            sys.path.insert(0, str(self.source_path))
        from vggt.utils.load_fn import load_and_preprocess_images

        frame_count = len(frame_paths)
        if anchor_frame < 0 or anchor_frame >= frame_count:
            raise ValueError("anchor_frame out of range")
        order = [anchor_frame, *[i for i in range(frame_count) if i != anchor_frame]]
        inverse = [order.index(i) for i in range(frame_count)]
        permuted_paths = [frame_paths[i] for i in order]
        images = load_and_preprocess_images(permuted_paths).to(self.device)
        query = query_xy_vggt.to(device=self.device, dtype=torch.float32)
        with _autocast(self.device, self.dtype):
            predictions = model(images, query_points=query)
        track = predictions["track"].squeeze(0).detach().cpu().float()[inverse]
        vis = predictions["vis"].squeeze(0).detach().cpu().float()[inverse]
        conf = predictions["conf"].squeeze(0).detach().cpu().float()[inverse]
        return track, vis, conf


def attach_tracks(
    geometry: VGGTGeometry,
    *,
    track_xy: torch.Tensor,
    track_vis: torch.Tensor,
    track_conf: torch.Tensor,
    track_anchor_frames: torch.Tensor,
    track_query_token_ids: torch.Tensor,
) -> VGGTGeometry:
    return VGGTGeometry(
        extrinsic_cw=geometry.extrinsic_cw,
        intrinsic=geometry.intrinsic,
        depth=geometry.depth,
        depth_conf=geometry.depth_conf,
        world_points_head=geometry.world_points_head,
        world_points_unproj=geometry.world_points_unproj,
        point_conf=geometry.point_conf,
        frame_transforms=geometry.frame_transforms,
        track_xy=track_xy,
        track_vis=track_vis,
        track_conf=track_conf,
        track_anchor_frames=track_anchor_frames,
        track_query_token_ids=track_query_token_ids,
    )


def _resolve_local_weight(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("model.safetensors", "model.pt", "pytorch_model.bin"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No VGGT weight file found under {path}")


def _default_frame_transforms(frame_paths: list[str], image_size: int) -> tuple[FrameTransform, ...]:
    transforms = []
    for frame_id, path in enumerate(frame_paths):
        with Image.open(path) as image:
            transforms.append(make_frame_transform(frame_id, image.size, (image_size, image_size), (image_size, image_size)))
    return tuple(transforms)


def _dtype_from_name(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).replace("torch.", "")
    if name == "bf16":
        name = "bfloat16"
    return getattr(torch, name)


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)
