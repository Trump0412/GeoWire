# GeoWire：Codex 实现脚手架与代码执行规范

> **用途**：本文件是 Codex 的直接实施合同。Codex 必须以 `GeoWire_Master_Design.md` 为最高理论约束，以本文件为唯一的工程拆解、接口定义、验收标准和执行顺序。
> **版本**：`v0.2`
> **日期**：2026-07-05
> **目标仓库名**：`geowire`
> **主模型**：`Qwen3-VL-2B-Instruct` + frozen `VGGT-1B`
> **第一阶段目标**：先证明 **正确几何图能够恢复与隔离语义 token**；在此以前不得启动大规模 spatial QA SFT。

> **v0.2 training override**：`GeoWire_Training_Protocol.md` 覆盖本文件旧版 TIP 内容。Codex 必须执行：Phase 1 仅优化 `L_rec + λ_sub L_sub (+ optional low-weight L_keep)`；`Non-edge isolation` 仅作为诊断，不得作为训练 loss；support substitution 必须在 masked target 上执行；Phase 2 按 `3 QA batches : 1 TIP batch` 交替训练。


---

# 0. 总约束：Codex 不得擅自改变的设计

GeoWire 的唯一主线是：

\[
\boxed{
\text{VGGT geometry} \rightarrow P_G \rightarrow \text{semantic token transport} \rightarrow \text{Qwen merger / LLM}
}
\]

其中：

- VGGT 只提供冻结的 **geometry control plane**；
- Qwen vision encoder 只提供语义 payload；
- `P_G` 是由 track、visibility、pointmap、camera、depth 与几何认证构成的稀疏 correspondence operator；
- GeoWire 只沿 `P_G` 传递 **Qwen semantic tokens**；
- geometry feature、depth token、pointmap token、VGGT hidden state **不得**作为 value 拼接、add、concat 或 cross-attention 注入 Qwen；
- 不实现 geometry bank、continuity bank、entity memory、object ID、query router、token competition、文本条件边权或语义相似度边权；
- 主模型只在 **Qwen vision encoder 与 original visual merger 之间**插入两层 GeoWire transport；
- 任何没有通过对应单测与图可视化的边，不允许进入训练图。

如果某个实现需求与上述原则冲突，Codex 必须停止该分支，记录冲突原因，而不是临时加入新的融合模块。

---

# 1. 实施策略：按闭环拆分，不一次实现完整模型

## 1.1 实施里程碑

| Milestone | 名称 | 产物 | 允许继续的条件 |
|---|---|---|---|
| M0 | 环境与模型 smoke test | 环境报告、VGGT/Qwen 最小前向 | 版本、权重、GPU 和输入输出 shape 全部记录 |
| M1 | Coordinate Contract | Qwen/VGGT/raw 三坐标映射、解析单测 | patch center 到 raw pixel 往返误差 < 0.5 px |
| M2 | VGGT cache 与 graph builder | per-clip geometry cache、COO graph、可视化 | 人工检查 100 个 clip，错边率可接受 |
| M3 | Sparse GeoWire block | 稀疏 transport、随机图对照、数值单测 | sparse 与 dense 参考实现一致 |
| M4 | TIP pretraining | masked recovery / substitution / isolation | full graph 明显优于 random/shuffled graph |
| M5 | Qwen3-VL bridge | encoder → GeoWire → merger → LLM | `alpha=0` 与原模型 logits 保持一致 |
| M6 | Spatial QA SFT | LoRA 训练、统一 evaluator | VSI-Debiased/MMSI 机制子集有稳定收益 |
| M7 | 完整对比与论文证据 | benchmark 表、诊断、可视化 | 所有主张可被消融支撑 |

**禁止跳过 M1–M4 直接进行 M6。**

## 1.2 每个 Milestone 的提交要求

每完成一个 milestone：

1. 代码可从全新环境运行；
2. 增加单测；
3. 生成一个最小命令行例子；
4. 写入 `docs/EXPERIMENT_LOG.md`；
5. 保存 `runs/<run_id>/config.yaml`、`git_state.json`、`env.json`、`metrics.jsonl`；
6. 在未通过验收前不得把下一阶段结果写入论文或汇报。

---

# 2. 仓库结构

```text
geowire/
├── README.md
├── pyproject.toml
├── environment.yml
├── requirements-vggt.txt
├── requirements-train.txt
├── Makefile
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── phase0_graph.yaml
│   ├── phase1_tip.yaml
│   ├── phase2_sft.yaml
│   ├── eval_vsi.yaml
│   ├── eval_mmsi.yaml
│   ├── eval_viewspatial.yaml
│   └── data/
│       ├── debug_toy.yaml
│       ├── llava_hound.yaml
│       ├── spar.yaml
│       └── benchmark_only.yaml
├── geowire/
│   ├── __init__.py
│   ├── cli.py
│   ├── types.py
│   ├── constants.py
│   ├── utils/
│   │   ├── distributed.py
│   │   ├── io.py
│   │   ├── logging.py
│   │   ├── reproducibility.py
│   │   ├── tensor.py
│   │   └── visualization.py
│   ├── data/
│   │   ├── manifest.py
│   │   ├── clip_dataset.py
│   │   ├── tip_dataset.py
│   │   ├── spatial_sft_dataset.py
│   │   ├── collate.py
│   │   ├── leakage_audit.py
│   │   └── samplers.py
│   ├── geometry/
│   │   ├── transforms.py
│   │   ├── qwen_layout.py
│   │   ├── vggt_provider.py
│   │   ├── vggt_cache.py
│   │   ├── track_queries.py
│   │   ├── projective_edges.py
│   │   ├── certify_edges.py
│   │   ├── graph_builder.py
│   │   ├── graph_io.py
│   │   └── graph_stats.py
│   ├── models/
│   │   ├── sparse_transport.py
│   │   ├── geowire.py
│   │   ├── qwen3vl_bridge.py
│   │   ├── qwen3vl_vision.py
│   │   ├── lora.py
│   │   └── dummy_backends.py
│   ├── objectives/
│   │   ├── tip.py
│   │   ├── qa.py
│   │   └── metrics.py
│   ├── training/
│   │   ├── common.py
│   │   ├── pretrain_tip.py
│   │   ├── train_sft.py
│   │   ├── checkpoints.py
│   │   └── scheduler.py
│   ├── evaluation/
│   │   ├── diagnostics.py
│   │   ├── eval_vsi.py
│   │   ├── eval_mmsi.py
│   │   ├── eval_viewspatial.py
│   │   ├── protocol.py
│   │   └── report.py
│   └── visualization/
│       ├── edges.py
│       ├── transport.py
│       ├── interventions.py
│       └── case_study.py
├── scripts/
│   ├── inspect_environment.py
│   ├── inspect_qwen3vl.py
│   ├── cache_vggt.py
│   ├── build_graphs.py
│   ├── verify_coordinate_contract.py
│   ├── verify_graph.py
│   ├── train_tip.py
│   ├── train_sft.py
│   ├── evaluate.py
│   ├── audit_leakage.py
│   └── export_case_study.py
├── tests/
│   ├── conftest.py
│   ├── test_transforms.py
│   ├── test_qwen_layout.py
│   ├── test_vggt_provider.py
│   ├── test_track_anchor.py
│   ├── test_projective_edges.py
│   ├── test_certify_edges.py
│   ├── test_graph_builder.py
│   ├── test_sparse_transport.py
│   ├── test_tip_losses.py
│   ├── test_qwen_bridge_parity.py
│   ├── test_permutation.py
│   └── test_leakage_audit.py
├── docs/
│   ├── MASTER_DESIGN.md
│   ├── CODEX_IMPLEMENTATION_SCAFFOLD.md
│   ├── EXPERIMENT_LOG.md
│   ├── PAPER_EVIDENCE.md
│   ├── DATA_CARD.md
│   └── DECISIONS.md
├── third_party/
│   └── vggt/                 # git submodule or pinned source checkout; never edit upstream files
├── assets/
│   └── toy_scene/
└── runs/
```

`docs/MASTER_DESIGN.md` 必须是顶层设计文档的副本或软链接。`CODEX_IMPLEMENTATION_SCAFFOLD.md` 必须是本文件副本或软链接。

---

# 3. 环境、版本与可复现性

## 3.1 Python 环境

优先使用 Python 3.10，CUDA 与 PyTorch 版本必须匹配服务器驱动。初始建议：

```yaml
name: geowire
channels: [pytorch, nvidia, conda-forge]
dependencies:
  - python=3.10
  - pip
  - pytorch>=2.5
  - torchvision
  - pytorch-cuda=12.4
  - pip:
      - transformers
      - accelerate
      - peft
      - deepspeed
      - safetensors
      - huggingface_hub
      - qwen-vl-utils
      - einops
      - hydra-core
      - omegaconf
      - pyarrow
      - pillow
      - opencv-python-headless
      - matplotlib
      - pytest
      - ruff
      - mypy
      - tensorboard
      - wandb
```

实际版本不能只写 `latest`。M0 完成后必须执行：

```bash
python scripts/inspect_environment.py --write runs/env_report.json
pip freeze > requirements-lock.txt
```

训练和评测必须保存 `requirements-lock.txt` 的 hash。

## 3.2 VGGT 固定版本

- 将官方 VGGT 仓库作为 `third_party/vggt` 固定到一个 commit；
- 在 `docs/DECISIONS.md` 记录 commit hash、checkpoint 名称与许可状态；
- GeoWire 不修改 `third_party/vggt` 内代码；所有包装逻辑置于 `geowire/geometry/vggt_provider.py`；
- 默认 checkpoint：`facebook/VGGT-1B`；若使用 commercial checkpoint，须明确记录许可证和权重名。

官方 VGGT 的 `VGGT.forward(images, query_points)` 会返回 pose、depth、depth confidence、world points，并可在指定 query points 时返回 tracks、visibility 和 confidence。当前官方代码要求 query points 的形状为 `[B, N, 2]`，且 tracker 将输入 sequence 的第 0 帧视为 query reference frame。因此任意 source frame 的 track 不能被假设为“一个前向自动得到”，必须显式处理 anchor frame。

## 3.3 Qwen3-VL 固定版本

- 主 checkpoint：`Qwen/Qwen3-VL-2B-Instruct`；
- `qwen-vl-utils`、`transformers` 必须固定到 M0 验证通过的版本；
- Phase 0/1 用 **多图像序列**，每帧作为独立 image placeholder 输入，而不是 Qwen video mode；
- 原因：多图像模式能够保留“每帧 token layout”的确定映射，避免视频时间压缩导致 Qwen token 与 VGGT frame/pixel 对齐不明确；
- Phase 2 可以对比 Qwen 原生 video mode，但只有在 TokenLayout 已能精确支持 temporal merge 后才允许启用。

每一帧前插入轻量文本标签，例如：

```text
[Frame 0 | timestamp 0.00s] <image>
[Frame 1 | timestamp 0.50s] <image>
...
Question: ...
```

标签只用于保留时间顺序语义，不参与 `P_G` 构图。

---

# 4. 核心数据合同（必须先实现 dataclass）

在 `geowire/types.py` 中定义以下不可变 dataclass。所有核心模块必须使用这些对象，而不是无语义的字典传参。

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import torch


@dataclass(frozen=True)
class Affine2D:
    """Homogeneous transform mapping xy pixel centers between two 2D coordinate systems."""
    matrix: torch.Tensor  # [3, 3], float64, source_xy -> target_xy


@dataclass(frozen=True)
class FrameTransform:
    frame_id: int
    raw_size_wh: tuple[int, int]
    qwen_size_wh: tuple[int, int]
    vggt_size_wh: tuple[int, int]
    raw_to_qwen: Affine2D
    qwen_to_raw: Affine2D
    raw_to_vggt: Affine2D
    vggt_to_raw: Affine2D
    valid_qwen_rect_xyxy: tuple[float, float, float, float]
    valid_vggt_rect_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class TokenLayout:
    """One flattened visual-token sequence with explicit per-token geometry."""
    num_frames: int
    hidden_size: int
    token_offsets: torch.Tensor      # [F + 1], int64; token slice for each frame
    frame_index: torch.Tensor        # [N], int64
    grid_row: torch.Tensor           # [N], int64
    grid_col: torch.Tensor           # [N], int64
    center_qwen_xy: torch.Tensor     # [N, 2], float32
    center_raw_xy: torch.Tensor      # [N, 2], float32
    center_vggt_xy: torch.Tensor     # [N, 2], float32
    valid: torch.Tensor              # [N], bool


@dataclass(frozen=True)
class VGGTGeometry:
    extrinsic_cw: torch.Tensor       # [F, 4, 4], camera-from-world, OpenCV convention
    intrinsic: torch.Tensor          # [F, 3, 3]
    depth: torch.Tensor              # [F, H_v, W_v]
    depth_conf: torch.Tensor         # [F, H_v, W_v]
    world_points_head: torch.Tensor  # [F, H_v, W_v, 3]
    world_points_unproj: torch.Tensor# [F, H_v, W_v, 3]
    point_conf: torch.Tensor         # [F, H_v, W_v]
    frame_transforms: tuple[FrameTransform, ...]
    # Optional anchor-dependent tracks, not necessarily populated for every frame.
    track_xy: Optional[torch.Tensor]        # [A, F, Q, 2] in VGGT pixel coords
    track_vis: Optional[torch.Tensor]       # [A, F, Q]
    track_conf: Optional[torch.Tensor]      # [A, F, Q]
    track_anchor_frames: Optional[torch.Tensor]  # [A], each anchor's original frame id
    track_query_token_ids: Optional[torch.Tensor]# [A, Q], flattened Qwen token ids


@dataclass(frozen=True)
class SparseGraph:
    """COO graph where destination receives weighted messages from source."""
    num_nodes: int
    dst: torch.Tensor       # [E], int64
    src: torch.Tensor       # [E], int64
    weight: torch.Tensor    # [E], float32; row-normalized over dst
    edge_type: torch.Tensor # [E], uint8 bitmask: SELF=1, TRACK=2, PROJ=4
    reproj_error: torch.Tensor  # [E], float32, nan for self
    cycle_error: torch.Tensor   # [E], float32, nan when unavailable
    visibility: torch.Tensor    # [E], float32
    confidence: torch.Tensor    # [E], float32


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    scene_id: str
    source_dataset: str
    frame_paths: tuple[str, ...]
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    split: Literal["train", "val", "test"]
    question: Optional[str] = None
    answer: Optional[str] = None
    task_type: Optional[str] = None
    static_view_permutation_allowed: bool = False
    cache_dir: Optional[str] = None
```

## 4.1 坐标约定

严格使用下列约定：

- 原图、Qwen canvas、VGGT canvas 均用 **pixel-center** 坐标；左上第一个像素中心为 `(0.5, 0.5)`；
- 所有 `xy` 顺序是 `(u, v)`，即 `(x, y)`；
- VGGT camera extrinsic 使用官方定义的 **camera-from-world** OpenCV convention；
- 在任一函数 docstring 内注明输入坐标系、输出坐标系和单位；
- 禁止混用 `(row, col)` 与 `(x, y)`；只在 tensor indexing 时明确转换；
- `float64` 用于变换矩阵与解析验证；模型计算可转 `float32/bfloat16`。

---

# 5. 数据清单、cache 布局与泄漏审计

## 5.1 Manifest JSONL

每条 clip 一行。最小格式：

```json
{
  "clip_id": "dataset_scene_000123_clip_04",
  "scene_id": "dataset_scene_000123",
  "source_dataset": "llava_hound",
  "frame_paths": ["/abs/f0.jpg", "/abs/f1.jpg", "/abs/f2.jpg"],
  "frame_indices": [12, 18, 24],
  "timestamps_s": [0.4, 0.6, 0.8],
  "split": "train",
  "question": "...",
  "answer": "...",
  "task_type": "spatial_relation",
  "static_view_permutation_allowed": false
}
```

禁止仅以文件名去重。`scene_id` 必须用于 leakage audit。

## 5.2 每个 clip 的 cache 格式

```text
cache_root/
└── <clip_id>/
    ├── metadata.json
    ├── geometry.safetensors
    ├── token_layout.safetensors
    ├── graph_coo.npz
    ├── frame_transforms.json
    ├── qwen_input.json
    ├── source_record.json
    └── debug/
        ├── source_target_edges_frame0.png
        ├── projection_overlay_frame0.png
        ├── graph_stats.json
        └── contract_report.json
```

### `geometry.safetensors`

必须包含：

```text
extrinsic_cw             float32 [F,4,4]
intrinsic                float32 [F,3,3]
depth                    float32 [F,Hv,Wv]
depth_conf               float32 [F,Hv,Wv]
world_points_head        float32 [F,Hv,Wv,3]
world_points_unproj      float32 [F,Hv,Wv,3]
point_conf               float32 [F,Hv,Wv]
track_xy                 float32 [A,F,Q,2]  optional
track_vis                float32 [A,F,Q]    optional
track_conf               float32 [A,F,Q]    optional
track_anchor_frames      int64   [A]        optional
track_query_token_ids    int64   [A,Q]      optional
```

### `graph_coo.npz`

```text
dst, src                 int64 [E]
weight                   float32 [E]
edge_type                uint8 [E]
reproj_error             float32 [E]
cycle_error              float32 [E]
visibility               float32 [E]
confidence               float32 [E]
num_nodes                int64 [1]
```

## 5.3 Cache 版本签名

`metadata.json` 必须记录：

```json
{
  "cache_schema": "v1",
  "clip_id": "...",
  "vggt_repo_commit": "...",
  "vggt_checkpoint": "facebook/VGGT-1B",
  "qwen_checkpoint": "Qwen/Qwen3-VL-2B-Instruct",
  "transformers_version": "...",
  "qwen_vl_utils_version": "...",
  "vggt_preprocess_mode": "pad",
  "qwen_preprocess_spec": {"...": "..."},
  "graph_config_hash": "...",
  "created_at_utc": "..."
}
```

当任一关键字段变化时，旧 cache 必须视为不可复用。

---

# 6. Qwen token layout 与坐标对齐

这是整个项目最容易出错、优先级最高的部分。

## 6.1 Phase 0 输入策略

Phase 0、Phase 1、Phase 2 的默认输入均将一个 video clip 表达为 **有序多图像**，不调用 Qwen video placeholder。

原因：

1. 每帧可获得独立 token slice；
2. token 与原图 / VGGT 坐标的映射显式；
3. 避免 Qwen 视频 temporal merge 导致一个视觉 token 混合多个时刻；
4. GeoWire 首先验证“跨帧 patch correspondence”，不应让 Qwen 内部时间压缩干扰该验证。

## 6.2 `QwenFramePreprocessor`

实现文件：`geowire/geometry/qwen_layout.py`

职责：

1. 使用固定版本 Qwen processor 或 `qwen_vl_utils` 的官方 resize 逻辑得到每帧 Qwen canvas；
2. 记录从 raw image 到 Qwen canvas 的完整仿射变换；
3. 提供 `center_qwen_xy`、`center_raw_xy` 和 validity mask；
4. 不手写无法验证的“近似 Qwen resize”；必须调用官方预处理函数或使用其明确的同等实现；
5. 处理 padding 区：落在 padding 区的 token 标记 `valid=False`，不建立 track/projection 边。

建议接口：

```python
class QwenFramePreprocessor:
    def __init__(self, processor, *, frame_mode: str, min_pixels: int, max_pixels: int): ...

    def prepare_images(
        self,
        frame_paths: list[str],
    ) -> tuple[dict[str, torch.Tensor], tuple[FrameTransform, ...], dict]:
        """Return processor-ready tensors, exact transforms, and metadata."""


class QwenTokenLayoutBuilder:
    def __init__(self, model_config): ...

    def build(
        self,
        image_grid_thw: torch.Tensor,
        frame_transforms: tuple[FrameTransform, ...],
        visual_token_count: int,
    ) -> TokenLayout:
        """Derive visual-token centers from the actual model grid and merge settings."""
```

## 6.3 禁止硬编码 Qwen token 网格

不能假设：

- patch size 永远是 16；
- spatial merge 永远为 2；
- 每帧 token 数恒定；
- 视频 token 与图像 token layout 相同；
- 图像一定是方形。

必须从当前模型 / processor 输出中读取：

- `image_grid_thw`；
- vision config 的 patch size；
- spatial merge size；
- 实际 `visual_token_count`；
- 每个图像 placeholder 对应 token range。

`inspect_qwen3vl.py` 必须打印并保存这些信息。

## 6.4 `TokenLayout` 验收

对任意随机 clip：

1. 绘制全部 token center 到 raw image；
2. 画出 Qwen valid image rectangle；
3. 验证每个 valid token center 在 raw image 内；
4. 映射 raw → Qwen → raw；最大误差 < 0.5 px；
5. 对 4 种宽高比图像都通过：`16:9`、`9:16`、`4:3`、极端长图。

---

# 7. VGGT provider 与 arbitrary-anchor tracking

## 7.1 官方接口封装

实现：`geowire/geometry/vggt_provider.py`

```python
class VGGTProvider(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        dtype: torch.dtype,
        preprocess_mode: str = "pad",
    ) -> None: ...

    @torch.inference_mode()
    def infer_geometry(
        self,
        frame_paths: list[str],
    ) -> VGGTGeometry:
        """Canonical order forward: camera/depth/point maps and transforms."""

    @torch.inference_mode()
    def track_from_anchor(
        self,
        geometry: VGGTGeometry,
        anchor_frame: int,
        query_xy_vggt: torch.Tensor,
        *,
        mode: Literal["full_permute", "reordered_features"] = "full_permute",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tracks [F,Q,2], vis [F,Q], conf [F,Q] in original frame order."""
```

## 7.2 必须处理的 VGGT track 限制

当前官方 tracker 使用 sequence 的第 0 帧作为 query reference。也就是说：

```text
model(images, query_points)  # query_points 属于第 0 帧
```

不允许错误地把任意 frame `t` 的 point 直接送给 canonical sequence 的 tracker。

### 正确做法：anchor permutation

对 source frame `a`：

```text
permutation = [a] + [f for f in range(F) if f != a]
images_perm = images[permutation]
query_points_perm = source points in frame a coordinates
tracks_perm = VGGT(images_perm, query_points_perm)
tracks_original = inverse_permute(tracks_perm, permutation)
```

输出必须恢复到 original frame index。

### 优化做法：reordered feature reuse（后置，不可先实现）

后续可以尝试：先获得 canonical `aggregated_tokens_list`，再将 sequence 维度与 `images` 同步 reorder，调用 `track_head`。但是该方案必须与 `full_permute` 结果逐 clip 比较：

```text
track coordinate median error < 1.0 VGGT pixel
visibility correlation > 0.98
confidence correlation > 0.98
```

任何一项不通过，则禁止使用 feature reuse；Phase 0–1 使用 `full_permute` 正确性优先。

## 7.3 Track anchor 策略

配置：

```yaml
graph:
  track_anchor_strategy: uniform_m   # first | uniform_m | all
  num_track_anchors: 3
```

默认 `uniform_m=3`：选择首帧、中帧、末帧。

- `first`：只用于 smoke test；
- `uniform_m`：Phase 0/1 默认，控制离线成本；
- `all`：小数据集、高质量分析、DSR 动态扩展使用。

每个 anchor 查询其对应 frame 内的所有 valid Qwen token centers，或使用间隔采样的 token centers；默认从每个 Qwen token 查询一个中心。

## 7.4 point map 默认来源

官方 README 指出由 depth + camera unprojection 构建的 3D points 通常比 point head 更准确。默认使用：

```python
world_points = unproject_depth_map_to_point_map(depth, extrinsic_cw, intrinsic)
```

但必须同时缓存：

- `world_points_unproj`：默认 graph construction；
- `world_points_head`：诊断、消融、错误分析。

不可在论文中声称其为新的 pointmap；它只是对官方输出的更稳健使用方式。

---

# 8. Correspondence graph builder

实现：

```text
geowire/geometry/
├── track_queries.py
├── projective_edges.py
├── certify_edges.py
└── graph_builder.py
```

## 8.1 图的方向与索引

flatten token index：

```text
node_id = token_layout.token_offsets[frame_id] + local_token_id
```

图的存储语义固定为：

```text
(dst, src, weight)
```

表示 destination token `dst` 从 source token `src` 接收 `weight * H[src]`。

禁止在不同模块中混用 `(src,dst)` 与 `(dst,src)`。

## 8.2 Track edge 构建

对每个 anchor `a` 与其 query token `i=(a,p)`：

1. 取 token center `center_vggt_xy[i]`；
2. `track_from_anchor(a, query_xy)` 得到每一帧的 track xy、vis、conf；
3. 对 target frame `s`：
   - `vis[s,q] >= vis_threshold`；
   - `conf[s,q] >= track_conf_threshold`；
   - target xy 在 `valid_vggt_rect` 内；
   - 通过 `vggt -> raw -> qwen` 映射到 target Qwen grid；
   - 用 nearest valid target token 得到 node `j=(s,q)`；
4. 添加 `dst=i, src=j` 与 `dst=j, src=i` 两个方向边，除非配置指定 causal graph；
5. edge metadata 保存 anchor、target frame、track coordinate、vis、conf。

`track_from_anchor` 产生的是锚帧点在其他帧的位置。它不是对象 ID；不可跨 clip 连接。

## 8.3 Projective edge 构建

对每个 source token `i=(t,p)`：

1. 将 `center_vggt_xy[i]` 双线性采样 `world_points_unproj[t]` 得到 `X_world`；
2. 使用 target camera `K_s`, `T_cw_s`：

\[
X_{cam}=R_{cw}X_{world}+t_{cw}
\]

\[
(u,v,1)^T \sim K_s X_{cam}
\]

3. 若 `z_cam <= 0` 或投影在 target valid rect 外，拒绝；
4. 双线性采样 target `depth[s]`、`depth_conf[s]`；
5. 计算相对深度残差：

\[
e_{depth}=\frac{|D_s(u,v)-z_{cam}|}{\max(z_{cam},\epsilon)}
\]

6. 若 `e_depth < depth_rel_threshold` 且 `depth_conf > threshold`，投影位置映射到 target Qwen token `j`；
7. 加入有向边 `i <- j` 与 `j <- i`，但反向方向必须依据其自身的投影检查；禁止假定投影边天然对称。

## 8.4 Edge certification

实现函数：

```python
def certify_edge(
    *,
    track_conf: float | None,
    visibility: float | None,
    reproj_error_px: float | None,
    depth_rel_error: float | None,
    cycle_error_px: float | None,
    thresholds: EdgeThresholds,
) -> tuple[bool, float, dict[str, float]]:
    ...
```

默认硬过滤：

```yaml
graph:
  visibility_min: 0.50
  track_conf_min: 0.50
  depth_conf_min: 1.00
  reproj_error_px_max: 8.0
  depth_rel_error_max: 0.08
  cycle_error_px_max: 6.0
  topk_cross_frame: 4
  self_loop_weight: 1.0
```

上面数值仅为 Phase 0 初值。必须在 debug subset 上画 precision-coverage 曲线后再更新。

### 权重

不允许任何文本、语义 embedding 或可学习 token gate 进入边权。

```python
def geometric_edge_weight(
    track_conf: Tensor,
    visibility: Tensor,
    reproj_error_px: Tensor,
    cycle_error_px: Tensor,
    tau_reproj: float,
    tau_cycle: float,
) -> Tensor:
    return (
        track_conf
        * visibility
        * torch.exp(-reproj_error_px / tau_reproj)
        * torch.exp(-cycle_error_px / tau_cycle)
    )
```

对仅 projective edge，`track_conf=1`、`visibility=1`；对无法计算 cycle 的边，必须明确配置是否拒绝或只采用其他认证项。

## 8.5 合并、去重、top-K 与自环

对同一个 `(dst, src)`：

```text
combined_weight = 1 - (1 - w_track) * (1 - w_proj)
edge_type = TRACK | PROJ
```

然后：

1. 仅在 cross-frame edge 中按 weight 取 top-K；
2. 强制为每个 valid token 加 `SELF` edge；
3. 行归一化，使每个 destination 的入边权重和为 1；
4. 无 valid cross-frame 邻居的 token 只保留 self-loop；
5. invalid/padding token 不作为 source 或 destination。

## 8.6 图统计与可视化

`graph_stats.py` 输出：

```text
num_nodes / valid_nodes / num_edges
mean/median incoming degree
track-vs-proj edge ratio
edge weight distribution
frame-gap distribution
visibility/confidence distribution
reprojection and cycle error distribution
per-frame coverage
isolated token fraction
```

`visualization/edges.py` 至少输出：

- source frame + target frame 连线图；
- track edge 与 projection edge 分色；
- 正确高置信、低置信拒绝、cycle 失败三类边；
- 叠加 Qwen patch grid；
- 10 个最佳、10 个最差、10 个随机 case。

---

# 9. Sparse GeoWire transport

## 9.1 输入输出

实现：`geowire/models/sparse_transport.py`

```python
class SparseTransport(torch.nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,      # [N, D] or [B, N, D]
        graph: SparseGraph,
    ) -> torch.Tensor:
        """Return P_G @ hidden using COO gather/scatter without materializing [N,N]."""
```

Batched ragged clips 首选实现方式：每个 batch 内将 node concatenate，给 graph node index 加 offset；不要 padding 成 `[B, max_N, max_N]` dense matrix。

## 9.2 数值实现

目标：

\[
M_i=\sum_{j\in\mathcal N(i)}P_G(i,j)H_j
\]

参考实现：

```python
def sparse_aggregate(hidden: torch.Tensor, graph: SparseGraph) -> torch.Tensor:
    # hidden: [N, D]
    src_values = hidden.index_select(0, graph.src)                     # [E, D]
    weighted = src_values * graph.weight.to(hidden.dtype).unsqueeze(-1)
    out = torch.zeros_like(hidden)
    out.index_add_(0, graph.dst, weighted)
    return out
```

初版用 `index_add_`，不要先上 `torch.sparse.mm`、自定义 CUDA 或图神经网络库。只有在 profiling 明确显示 bottleneck 后才替换。

## 9.3 GeoWire block

实现：`geowire/models/geowire.py`

```python
class GeoWireBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.value_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = torch.nn.Parameter(torch.zeros(()))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, graph: SparseGraph) -> torch.Tensor:
        # hidden: [N,D], one concatenated clip batch
        value = self.value_proj(self.norm(hidden))
        message = sparse_aggregate(value, graph)
        delta = self.output_proj(message)
        return hidden + self.alpha * self.dropout(delta)
```

`GeoWireTransport` 包含两个 block：

```python
class GeoWireTransport(torch.nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int = 2) -> None: ...
    def forward(self, hidden: torch.Tensor, graph: SparseGraph) -> torch.Tensor: ...
```

注意：

- 不引入 Q/K attention；
- 不引入 token-level gate；
- 不引入 text condition；
- 不改变 `graph.weight`；
- `alpha` 初始为零，确保插入 adapter 后从原模型行为连续起步；
- 两层仅允许在同一 `P_G` 上进行两跳 semantic transport；
- 训练时记录每层 `alpha`、transport delta norm、cross-frame message ratio。

## 9.4 必测性质

1. `alpha=0`：输出与输入完全相等；
2. self-loop only：输出为 `H + alpha * W_o(W_v(LN(H)))`，无跨帧泄漏；
3. COO 与 dense `P_G @ H` 参考实现误差 `<1e-6` float32；
4. permutation equivariance：同步重排 `H` 与 graph 节点编号，输出同样重排；
5. 无 cross-frame edge 时，添加一个不连通 distractor clip 不改变原 clip 输出；
6. mixed precision 下无 NaN/Inf。

---

# 10. Topology Intervention Pretraining（TIP）

TIP 只服务 GeoWire，不引入第二网络。

## 10.1 Phase 1 模型

```text
Qwen vision encoder (frozen)
        ↓
clean visual tokens H_clean
        ↓  mask / replace / distract
H_input + fixed P_G
        ↓
GeoWire × 2
        ↓
H_transport
        ↓
TIP losses
```

Phase 1 不运行 visual merger 和 language decoder，不进行 QA generation。

## 10.2 Intervention sampler

实现：`geowire/data/tip_dataset.py`

```python
@dataclass(frozen=True)
class TIPTargets:
    masked_node_ids: torch.Tensor
    equivalent_support_groups: list[torch.Tensor]
    distractor_clip_ids: list[str]
    permutation: Optional[torch.Tensor]
```

### A. Correspondence-Masked Semantic Recovery

选节点条件：

```text
valid token
+ 至少 1 条高置信 cross-frame edge
+ source neighbor 位于不同 frame
+ target frame 的 token 不是 padding
```

操作：

```python
H_input = H_clean.clone()
H_input[masked_node_ids] = 0.0
```

损失：

\[
\mathcal L_{rec}=1-\cos(\hat h_i,\operatorname{sg}(h_i^{clean}))
\]

只在 `masked_node_ids` 上求值。

### B. Equivalent Support Substitution

从一个 destination token 的多个经过认证的跨帧 source neighbor 中随机分出两个 support sets `A/B`。构建两个只在该 destination 邻接不同的 graph：

```python
P_A = restrict_sources(P_G, dst=i, allowed_sources=A)
P_B = restrict_sources(P_G, dst=i, allowed_sources=B)
```

计算：

\[
\mathcal L_{sub}=\|f(H,P_A)_i-f(H,P_B)_i\|_2^2
\]

只对至少有两个独立 frame source 的节点启用。

### C. Non-edge Isolation

将外观相似但图不连通的 clip / frame 拼入 batch。图中不添加连接到 target clip 的边。目标 clip 的输出应保持稳定。

初版不做复杂“相似检索”。先实现两种 distractor：

1. 同一视频远距离、未连接的帧；
2. batch 内其他 clip。

后续才加入基于 CLIP/Qwen token 相似度的 hard distractor。不能把该相似度写入 graph。

损失：

\[
\mathcal L_{iso}=\|f(H,P_G)-f([H;H_{dis}],P_G\oplus P_{dis})\|_2^2
\]

### D. 可选静态多图 permutation

只在 `static_view_permutation_allowed=True` 的样本开启。

## 10.3 总损失

```python
loss = (
    lambda_rec * loss_recovery
    + lambda_sub * loss_substitution
    + lambda_iso * loss_isolation
    + lambda_perm * loss_permutation
)
```

Phase 1 默认：

```yaml
tip:
  lambda_rec: 1.0
  lambda_sub: 0.25
  lambda_iso: 0.10
  lambda_perm: 0.0
  mask_ratio: 0.15
```

这些值不是论文结论，必须做小规模 grid search 并记录。

## 10.4 Phase 1 指标

```text
masked_token_cosine
masked_token_mse
retrieval_at_1 / retrieval_at_5
substitution_consistency_cosine
non_edge_leakage_l2
random_graph_gap
shuffled_graph_gap
edge_coverage
```

必须同时对比：

```text
full certified graph
self-loop only
random graph (same degree distribution)
shuffled graph (same edge weights, wrong endpoints)
track-only graph
projective-only graph
```

若 full graph 无法稳定胜过 random/shuffled graph，停止进入 Phase 2。

---

# 11. Qwen3-VL bridge：安全插入 GeoWire

这是 Phase 2 的唯一高风险工程点。

## 11.1 原则

不能依赖任意 forward hook 猜测视觉 token 位置。必须实现一个显式 bridge，在 Qwen 的视觉编码输出、视觉 token 被替换进语言 `inputs_embeds` 之前调用 GeoWire。

目标执行路径：

```text
processor / multi-image batch
       ↓
Qwen visual module
       ↓
visual token sequence H
       ↓
TokenLayout + SparseGraph
       ↓
GeoWire(H, P_G)
       ↓
Qwen original image-token replacement logic
       ↓
Qwen language model / generation
```

## 11.2 先写 inspection script

`script/inspect_qwen3vl.py` 必须在未改模型时打印：

```text
transformers version
model class
vision module path
language module path
merger / projector module path
visual output shape
image_grid_thw
image placeholder count
visual token count
all candidate module names and forward signatures
```

将输出保存到 `runs/qwen_inspection/<timestamp>.json`。

## 11.3 Bridge 的实施方法

创建：`geowire/models/qwen3vl_bridge.py`

```python
class Qwen3VLGeoWireForConditionalGeneration(torch.nn.Module):
    """Pinned-version bridge. It reproduces the installed Qwen3-VL image embedding path,
    inserting GeoWire exactly once after the vision encoder and before image embeddings are
    scattered into language inputs_embeds.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        geowire: GeoWireTransport,
        token_layout_builder: QwenTokenLayoutBuilder,
    ) -> None: ...

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        graph: SparseGraph | BatchedSparseGraph,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ):
        ...
```

实施要求：

1. 从锁定的 Transformers 源码复制 **最小必要** 的 Qwen3-VL conditional generation forward 路径；
2. 保留与原模型一致的图像 token replacement、position ids、rope/temporal position、attention mask、cache、generation 行为；
3. 仅在原始 visual output 到 language embedding replacement 之间调用：

```python
visual_embeds = self._run_original_vision(pixel_values, image_grid_thw)
layout = self.layout_builder.build(...)
visual_embeds = self.geowire(visual_embeds, graph)
inputs_embeds = self._run_original_visual_embedding_replacement(
    input_ids=input_ids,
    visual_embeds=visual_embeds,
    ...,
)
```

4. 不在原 Qwen language layers 中插入 GeoWire；
5. 不使用 monkey patch 修改 site-packages；
6. 保存被适配的 Transformers 源文件路径、commit/version、diff 摘要到 `docs/DECISIONS.md`；
7. 若 Qwen3-VL 当前实现无法干净抽取该位置，先完成 **Qwen2.5-VL compatibility prototype** 验证 bridge 结构，再回到 Qwen3-VL；不得把 Qwen2.5 原型混入主结果。

## 11.4 必须通过的 parity test

`tests/test_qwen_bridge_parity.py`：

```text
same model weights
same processor inputs
same random seed
GeoWire alpha = 0
```

要求：

- logits 最大绝对误差：fp32 `<1e-5`；bf16 `<2e-3`；
- loss 误差在同一容差；
- greedy generation 前若干 token 相同；
- visual token count 与 image placeholder token count 一致；
- 单图、多图、不同宽高比均通过。

未通过 parity test 时，禁止开启任何 LoRA 或 GeoWire 训练。

## 11.5 LoRA 目标

Phase 2：

```yaml
lora:
  enabled: true
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - language_model.*.self_attn.q_proj
    - language_model.*.self_attn.k_proj
    - language_model.*.self_attn.v_proj
    - language_model.*.self_attn.o_proj
    - language_model.*.mlp.gate_proj
    - language_model.*.mlp.up_proj
    - language_model.*.mlp.down_proj
    - visual_merger.*
```

实际 module path 必须由 `inspect_qwen3vl.py` 自动验证。不存在的名称不得静默忽略；脚本必须 fail-fast。

默认冻结 vision encoder。只有当 Phase 2 已证实 graph mechanism 有效、且 visual encoder LoRA 能带来严格对照收益时，才尝试 vision LoRA。

---

# 12. 训练实现

## 12.1 Phase 0：构图命令

```bash
python scripts/cache_vggt.py \
  --config configs/phase0_graph.yaml \
  data.manifest=/path/to/debug_manifest.jsonl \
  runtime.output_dir=runs/phase0_cache

python scripts/build_graphs.py \
  --config configs/phase0_graph.yaml \
  data.manifest=/path/to/debug_manifest.jsonl \
  cache.root=/path/to/cache \
  runtime.output_dir=runs/phase0_graph

python scripts/verify_coordinate_contract.py \
  --config configs/phase0_graph.yaml \
  data.manifest=/path/to/debug_manifest.jsonl

python scripts/verify_graph.py \
  --config configs/phase0_graph.yaml \
  data.manifest=/path/to/debug_manifest.jsonl \
  visualization.num_cases=100
```

## 12.2 Phase 1：TIP pretraining

`geowire/training/pretrain_tip.py` 负责：

1. 加载 cache / graph；
2. 通过 frozen Qwen vision backend 得到 `H_clean`；
3. 生成 TIP interventions；
4. 运行 GeoWire；
5. 计算 TIP loss；
6. 输出 mechanism metrics；
7. 保存只包含 GeoWire 权重的 checkpoint。

建议命令：

```bash
torchrun --nproc_per_node=6 scripts/train_tip.py \
  --config configs/phase1_tip.yaml \
  data.manifest=/path/to/train_manifest.jsonl \
  data.cache_root=/path/to/cache \
  runtime.output_dir=runs/tip_qwen3vl2b_k4
```

初始配置：

```yaml
seed: 3407
model:
  qwen_checkpoint: Qwen/Qwen3-VL-2B-Instruct
  freeze_vision: true
  geowire_blocks: 2
  geowire_dropout: 0.0
geometry:
  vggt_checkpoint: facebook/VGGT-1B
  use_cached_graph: true
train:
  precision: bf16
  gradient_checkpointing: false
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  global_batch_target: 96
  lr: 2.0e-4
  weight_decay: 0.05
  warmup_ratio: 0.05
  max_steps: 30000
  log_every: 20
  save_every: 1000
```

## 12.3 Phase 2：Spatial SFT

`geowire/training/train_sft.py`：

```text
VGGT cache / P_G             frozen / loaded from disk
Qwen vision encoder          frozen
GeoWire                       trainable
visual merger                 LoRA or selected trainable layers
Qwen language decoder         LoRA
```

训练损失：

```python
loss = qa_loss + tip_weight * tip_loss
```

Phase 2 不应把 TIP 权重设为 0；建议从 `0.1–0.3` 搜索。

```yaml
train:
  lr_geowire: 1.0e-4
  lr_lora: 2.0e-5
  max_steps: 12000
  warmup_ratio: 0.03
  tip_weight: 0.2
  use_deepspeed_zero2: true
```

## 12.4 Checkpoint 内容

每个 checkpoint 保存：

```text
geowire_adapter.safetensors
lora_adapter/
trainer_state.json
config_resolved.yaml
base_model_manifest.json
graph_config.json
metrics.json
```

不保存 VGGT 和完整 Qwen 权重副本。

---

# 13. Baseline 实现优先级

## 13.1 先实现的内部受控基线

| ID | 实现 | 目的 |
|---|---|---|
| C0 | Vanilla Qwen3-VL | 无几何下限 |
| C1 | Raw geometry feature fusion | `H + MLP([point/depth/conf])`，仅作为 content-fusion 对照 |
| C3 | Dense temporal attention + geometry bias | 允许所有帧 token 通信，仅加 fixed geometric bias |
| C4 | GeoWire random graph | 验证正确图的重要性 |
| C5 | GeoWire shuffled graph | 验证 endpoints 而非 degree/weight 的作用 |
| C6 | GeoWire track-only | 动态 correspondence 贡献 |
| C7 | GeoWire projection-only | 静态几何 correspondence 贡献 |
| C8 | GeoWire full | 主模型 |

### `Raw geometry feature fusion` 的边界

这是公平对照，不是主模型。只允许在 C1 中使用：

```text
semantic token + projected geometry feature -> MLP -> token
```

不可复用该模块到 GeoWire full。

## 13.2 外部方法

外部已发表方法以统一协议实跑为目标：

```text
B0 Qwen3-VL-2B / 4B
B1 VG-LLM
B2 GeoThinker
B3 GeoSR
B4 GeoVR
```

若官方代码无法在同一数据、同一帧采样、同一 evaluator 下运行，报告时必须分为：

```text
Reproduced under our protocol
Reported by original paper
```

不得将两类数值混在同一个结论列。

---

# 14. Benchmark 封装与评测协议

## 14.1 统一输入协议

`geowire/evaluation/protocol.py` 定义：

```python
@dataclass(frozen=True)
class FrameSamplingProtocol:
    num_frames: int
    strategy: Literal["uniform", "centered_uniform", "benchmark_official"]
    include_first_last: bool
    max_frames: int

@dataclass(frozen=True)
class EvaluationProtocol:
    benchmark_name: str
    sampling: FrameSamplingProtocol
    prompt_template_id: str
    answer_normalizer_id: str
    model_generation_kwargs: dict
```

所有方法必须记录：

- 输入帧数；
- 采样策略；
- 图像分辨率 / token budget；
- prompt；
- generation 参数；
- evaluator version；
- 是否使用 benchmark 官方 frame selection。

## 14.2 主评测顺序

1. VSI-Bench-Debiased；
2. MMSI-Bench；
3. ViewSpatial-Bench；
4. 原始 VSI-Bench；
5. DSR-Bench（Phase 3）。

## 14.3 Diagnostics

`evaluation/diagnostics.py` 必须独立于 QA accuracy 输出：

```text
masked_token_cosine
transport_retrieval_at_1
transport_retrieval_at_5
substitution_consistency
non_edge_leakage
random_graph_gap
shuffled_graph_gap
frame_count_scaling
edge_certification_coverage
```

论文中不得只给主表总分；诊断必须作为主张依据。

---

# 15. 关键单元测试清单

## 15.1 几何与坐标

`test_transforms.py`

- identity transform；
- resize only；
- pad only；
- crop only；
- raw → qwen → raw 误差；
- raw → vggt → raw 误差。

`test_projective_edges.py`

- synthetic identity camera + planar depth；
- 两相机平移下的解析投影；
- `z_cam <= 0` 必须拒绝；
- depth 不一致必须拒绝；
- padding region 必须拒绝。

`test_track_anchor.py`

- 对 anchor=0、anchor=middle、anchor=last，输出都恢复到 original frame order；
- full-permute track 与保存的 reference 对齐；
- source anchor frame 的 track 坐标等于 query point（容差内）。

`test_certify_edges.py`

- visibility / conf / depth / cycle 中任一硬阈值失败，edge 被拒绝；
- 合法边 weight 正且有限；
- 组合边 weight 不超过 1。

## 15.2 图与 transport

`test_graph_builder.py`

- 每个 valid node 有 self loop；
- 每个 destination 入边和为 1；
- invalid token 不在图中；
- top-K 仅对 cross-frame edge 计数；
- edge merge 与 type bitmask 正确。

`test_sparse_transport.py`

- sparse gather/scatter 与 dense matrix 对齐；
- `alpha=0` identity；
- batch concatenate offset 正确；
- permutation equivariance；
- disconnected distractor isolation。

## 15.3 Qwen bridge

`test_qwen_bridge_parity.py`

- `alpha=0` logits parity；
- 单图、多图、不同宽高比；
- image placeholders 与 token layout slice 数量一致；
- generation cache 无 crash。

## 15.4 数据与泄漏

`test_leakage_audit.py`

- 同一 `scene_id` 不能同时出现在 train 与 benchmark test；
- scene id 缺失时 fail-fast；
- manifest hash 可复现；
- report 中包含已排除 scene 数量。

---

# 16. Debug 与失败处理原则

## 16.1 错边优先级高于训练 loss

一旦可视化显示 track/projection edge 经常落在错误物体：

```text
停止训练
→ 检查坐标变换
→ 检查 anchor frame 置换
→ 检查 camera convention
→ 检查 depth sampling / z-buffer
→ 调阈值并重新 cache
```

不要先尝试降低学习率、加 gate、改 LoRA rank 或堆更多数据。

## 16.2 若 random graph 与 full graph 性能相近

说明模型没有使用正确拓扑或 graph 质量不够。

检查顺序：

1. `alpha` 是否仍接近 0；
2. self-loop 是否权重过大；
3. masked recovery 是否真的遮掉 target semantic token；
4. graph 是否过密；
5. random graph 是否意外保留原邻居；
6. Qwen visual token 与 graph node 是否错位；
7. TIP 权重是否过小。

禁止用更大 LLM 掩盖问题。

## 16.3 若 full graph 比 self-loop 更差

说明错误边、过平滑或 graph semantic transport 与 task 不匹配。

处理顺序：

1. 降低 `K`：8 → 4 → 2；
2. 提高 edge certification 阈值；
3. 仅启用 track 或仅启用 projection 分析；
4. 检查两层 block 是否应退为一层；
5. 检查 alpha 是否过大；
6. 在 masked recovery 而非 QA 上先定位。

## 16.4 若 Qwen bridge parity 失败

不得继续训练。必须回到 pinned Transformers 模型源码，逐项比较：

```text
vision output
visual token ordering
placeholder mask
embedding scatter
position ids / rope ids
attention mask
cache position
```

不能通过 forward hook “绕过去”。

---

# 17. 配置文件模板

## 17.1 `configs/phase0_graph.yaml`

```yaml
defaults:
  - base
  - data: debug_toy

seed: 3407

geometry:
  vggt_checkpoint: facebook/VGGT-1B
  vggt_preprocess_mode: pad
  dtype: bfloat16
  track_anchor_strategy: uniform_m
  num_track_anchors: 3
  use_unprojected_points: true

qwen:
  checkpoint: Qwen/Qwen3-VL-2B-Instruct
  input_mode: ordered_images
  min_pixels: 262144
  max_pixels: 1048576

graph:
  topk_cross_frame: 4
  self_loop_weight: 1.0
  visibility_min: 0.50
  track_conf_min: 0.50
  depth_conf_min: 1.00
  reproj_error_px_max: 8.0
  depth_rel_error_max: 0.08
  cycle_error_px_max: 6.0
  tau_reproj: 4.0
  tau_cycle: 4.0

cache:
  root: /path/to/geowire_cache
  overwrite: false

visualization:
  num_cases: 100
  save_overlay: true

runtime:
  output_dir: runs/phase0_graph
```

## 17.2 `configs/phase1_tip.yaml`

```yaml
defaults:
  - phase0_graph

model:
  freeze_vision: true
  geowire_blocks: 2
  geowire_dropout: 0.0

tip:
  enabled: true
  mask_ratio: 0.15
  min_cross_frame_neighbors: 1
  lambda_rec: 1.0
  lambda_sub: 0.25
  lambda_iso: 0.10
  lambda_perm: 0.0

train:
  precision: bf16
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  max_steps: 30000
  lr: 0.0002
  weight_decay: 0.05
  warmup_ratio: 0.05
  num_workers: 8
  log_every: 20
  eval_every: 500
  save_every: 1000

runtime:
  output_dir: runs/phase1_tip
```

## 17.3 `configs/phase2_sft.yaml`

```yaml
defaults:
  - phase1_tip

model:
  qwen_bridge: true
  freeze_vision: true
  train_visual_merger_lora: true
  train_language_lora: true

lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: auto_from_inspection

tip:
  lambda_rec: 0.20
  lambda_sub: 0.05
  lambda_iso: 0.02

train:
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  max_steps: 12000
  lr_geowire: 0.0001
  lr_lora: 0.00002
  weight_decay: 0.01
  warmup_ratio: 0.03
  deepspeed: configs/deepspeed_zero2.json

runtime:
  output_dir: runs/phase2_sft
```

---

# 18. Codex 执行顺序（必须逐项完成）

## Step 1：初始化仓库

- 创建目录；
- 拷贝顶层设计文档；
- 添加 `pyproject.toml`、环境文件、lint/test 配置；
- 实现 `scripts/inspect_environment.py`；
- 运行 `pytest` 空测试与 `ruff check`。

**交付**：项目可安装、可测试、可输出环境报告。

## Step 2：实现 synthetic toy scene

在 `assets/toy_scene/` 自动生成：

- 两个平面；
- 两个相机；
- 已知深度、相机、点图；
- 已知投影对应。

必须先在 toy scene 验证 `Affine2D`、projective edge、graph normalization 和 sparse transport。不要先依赖真实 VGGT。

**交付**：toy test 100% 通过、生成 PNG overlay。

## Step 3：实现 Qwen layout 与 coordinate contract

- 多图 processor；
- `TokenLayout`；
- raw/Qwen/VGGT transform；
- 4 种宽高比单测；
- token center overlay。

**交付**：`verify_coordinate_contract.py` 成功。

## Step 4：实现 VGGT wrapper 与 cache

- canonical geometry forward；
- anchor-permute tracking；
- depth unprojection；
- safetensors cache；
- 每 clip metadata。

**交付**：一个真实 8 帧 clip 的 cache 完整生成。

## Step 5：实现 graph builder 与可视化

- track / projection / self edges；
- certify / merge / top-K / normalize；
- graph stats 和 overlay；
- 100 clip audit shell。

**交付**：`graph_coo.npz`、audit 报告、top/bottom/random overlays。

## Step 6：实现 sparse GeoWire 与 TIP

- `SparseTransport`；
- `GeoWireBlock × 2`；
- recovery/substitution/isolation sampler 与损失；
- random/shuffled graph 对照。

**交付**：Phase 1 debug run，输出 diagnostics 曲线。

## Step 7：通过 Phase 1 门槛

仅当：

```text
full graph > random graph >? baseline
full graph > shuffled graph
non-edge leakage decreases
masked recovery improves
```

并且图可视化无系统错位时，继续。

## Step 8：实现 Qwen3-VL bridge

- inspection；
- pinned-source bridge；
- alpha=0 parity；
- LoRA targeting。

**交付**：单图和八图的 forward / generation smoke test。

## Step 9：Spatial QA SFT 与受控 ablation

按 C0/C1/C3/C4–C8 顺序推进；每次只改变一个变量。

## Step 10：外部 baseline、最终 benchmark 与论文证据

- 统一 protocol；
- scene-level leakage audit；
- 生成 paper tables / figures；
- 在 `PAPER_EVIDENCE.md` 为每个论文结论链接到 run id、config、commit 和图表。

---

# 19. 最终交付验收清单

Codex 完成后，仓库必须能提供：

```text
[ ] One-command environment smoke test
[ ] Toy geometry contract tests
[ ] Qwen/VGGT coordinate alignment reports
[ ] VGGT cache for a real clip
[ ] Sparse graph and visual overlays
[ ] Sparse transport unit tests
[ ] TIP Phase 1 run with random/shuffled controls
[ ] Qwen bridge alpha=0 parity test
[ ] Phase 2 SFT entrypoint
[ ] Unified evaluation entrypoint
[ ] Leakage audit
[ ] Reproducible run artifact directory
[ ] README with exact commands
```

本脚手架的完成标准不是“代码文件都存在”，而是：

> **从 raw frames 出发，能够可视化 VGGT 几何边；从 Qwen 语义 token 出发，能够在正确 graph 上完成 sparse transport；在干预任务上证明 transport 使用的是正确 correspondence；随后才把它接入 Qwen 的语言推理链。**

---

# 20. 任何代码审查时的最终问题

> **Does this implementation make geometry decide where semantic evidence may safely travel, or does it inject geometry content back into the model?**

若是后者，拒绝合入主分支。
