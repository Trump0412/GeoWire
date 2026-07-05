# GeoWire 训练协议：阶段、任务、损失与数据

> **状态：v0.2 — training plan frozen**  
> **日期：2026-07-05**  
> 本文件补充并覆盖 `GeoWire_Master_Design.md` 中旧版 TIP 损失与数据描述。  
> 对 Codex 而言，本文件优先级仅低于 Master Design 的“geometry as control plane”主约束。

---

## 0. 最终决定

GeoWire 主稿只采用 **两个实际优化阶段**，外加一个不训练的几何准备阶段：

```text
Phase 0  Geometry Contract & Offline Cache        无梯度
Phase 1  Topology Intervention Pretraining (TIP)  只训练 GeoWire
Phase 2  Spatial Instruction Tuning (SIT)         GeoWire + merger LoRA + LLM LoRA
Phase 3  GeoVR / long-video / DSR 扩展            可选，不属于主模型训练
```

不增加：

- 独立的 geometry distillation stage；
- camera/depth/pointmap prediction head；
- entity memory、geometry bank、query router；
- RL、GSPO、OPD；
- geometry feature concat 或多层 hidden feature fusion。

核心因果链保持不变：

```text
VGGT correspondence graph P_G
      → GeoWire semantic transport
      → Qwen multimodal reasoning
```

Phase 1 的作用是让 GeoWire 学会正确使用 `P_G`；Phase 2 的作用是让这个已验证的 transport 能力转化为空间问答能力。

---

# 1. Phase 0：Geometry Contract 与离线缓存

## 1.1 是否训练

否。VGGT 与 Qwen 全部冻结，不更新任何参数。

## 1.2 目标

建立可被信任的 `per-clip sparse correspondence graph`。此阶段通过前，后续训练的任何结果都没有解释价值。

## 1.3 输入数据

### A. 几何干净开发池

优先使用 `SPAR-7M-RGBD` 的 **train split**，从多视图或视频样本采样 8 帧 clip。

- 优先源：`Structured3D`、`RxR`；
- `ScanNet`、`ScanNet++` 仅在完成测试 scene 排除后使用；
- 利用 RGBD 版本的深度、位姿和内参做 **图质量审计**，但绝不把 GT 几何直接写入 `P_G`。

### B. 自然视频开发池

使用 `LLaVA-Video-178K` 的 LLaVA-Hound split 中连续、无明显镜头切换的短 clip。

该池只用于检查 VGGT 在真实视频上的 edge coverage、visibility 和动态误差；不要求 QA 标签。

## 1.4 缓存内容

对每个 clip 保存：

```text
- frame RGB paths and raw sizes
- Qwen/VGGT/raw coordinate transforms
- VGGT intrinsics / extrinsics
- depth / depth_confidence
- pointmap / point_confidence
- track_xy / visibility / track_confidence
- certified sparse edges: src, dst, weight, type, diagnostics
- graph statistics and quality flags
```

## 1.5 准入门槛

随机人工检查不少于 100 个 clip，且同时报告：

```text
track edge precision
projective edge precision
cross-frame edge coverage
cycle residual distribution
reprojection residual distribution
visibility-consistent edge ratio
```

仅质量合格 clip 才进入 Phase 1。默认过滤规则：

```yaml
graph_filter:
  min_cross_frame_edges_per_valid_node: 1
  min_clip_cross_frame_coverage: 0.10
  require_visibility: true
  require_depth_consistency: true
  require_cycle_check: true
  reject_hard_scene_cut: true
```

阈值以 precision–coverage 曲线决定，不能写死后不再校准。

---

# 2. Phase 1：Topology Intervention Pretraining

## 2.1 目的

不是教模型回答问题，而是检验并训练一个更原子的能力：

> 当当前帧的局部语义证据被移除时，GeoWire 能否仅通过经过认证的跨帧 correspondence，将同一局部三维区域的语义恢复回来？

这阶段是论文第二贡献的训练基础。

## 2.2 可训练参数

| 组件 | 状态 |
|---|---:|
| VGGT-1B | 冻结，离线缓存 |
| Qwen3-VL visual encoder | 冻结，在线 `no_grad` 提取 token |
| GeoWire `W_v, W_o, alpha` | 训练 |
| Qwen visual merger | 不运行 |
| Qwen LLM | 不运行 |

Phase 1 不使用文本问题、不运行生成、不训练 LoRA。

## 2.3 数据组成

Phase 1 只需要多帧图像，不需要高质量文本答案。

| 数据池 | 初始占比 | 作用 |
|---|---:|---|
| `SPAR-7M-RGBD` 合格 train clips | 70% | 相机/深度可审计，稳定训练 correspondence transport |
| `LLaVA-Hound` 合格自然视频 clips | 30% | 真实外观、相机运动、遮挡与动态扰动 |

初始规模建议：

```text
SPAR-derived clips: 180k–240k
LLaVA-Hound clips: 45k–60k
Total unique clips: 225k–300k
Frames per clip: 8
```

这些是上限和采样目标，不要求先下载或缓存 SPAR-7M 全部 700 万 QA。

## 2.4 TIP 任务 A：Correspondence-Masked Semantic Recovery

### 样本条件

选择目标节点 `i=(t,p)`，要求：

```text
- token 有效；
- 至少一条来自其它帧的 certified edge；
- target 可见；
- 至少一个 source 可见且 confidence 足够高。
```

### 干预

对输入 token 做局部遮挡：

\[
H^{mask}_i=0,
\qquad H^{mask}_{j\ne i}=H^{clean}_j.
\]

保留 source-to-target 的真实图边与 self-loop。由于 target value 被清零，self-loop 无法泄漏无遮挡内容。

### 目标

令 `h_i^+ = stopgrad(normalize(H_clean_i))`，GeoWire 输出为 `\hat h_i`。采用 InfoNCE 恢复损失：

\[
\mathcal L_{rec}
=-\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}
\log
\frac{\exp(\operatorname{sim}(\hat h_i,h_i^+)/\tau)}
{\sum_{j\in\mathcal C_i}\exp(\operatorname{sim}(\hat h_i,h_j^+)/\tau)}.
\]

其中候选集合 `\mathcal C_i` 包含同 clip 中几何不对应 token 与 batch 内 token。InfoNCE 比单纯 cosine 更能防止模型恢复出无区分性的平均表征。

## 2.5 TIP 任务 B：Masked Equivalent-Support Substitution

这是 Phase 1 的第二个真正训练任务。

对同一个被遮挡 target `i`，从来自至少两个不同帧的高置信 source 中构造不相交支持集：

\[
\mathcal N_G^A(i),\quad \mathcal N_G^B(i).
\]

分别运行两张只在 target 入边不同的图：

\[
\hat h_i^A=f(H^{mask},P_G^A),
\qquad
\hat h_i^B=f(H^{mask},P_G^B).
\]

两个输出都使用 `\mathcal L_{rec}` 对齐 clean target，同时使用：

\[
\mathcal L_{sub}
=1-\cos(\hat h_i^A,\hat h_i^B).
\]

**重要修正**：替换任务必须在 target 被遮挡时执行。若保留 clean self-loop，`L_sub` 可以被当前帧 token 直接满足，训练会退化。

## 2.6 可选稳定器：Direct-Observation Preservation

它不是论文贡献，只是防止 GeoWire 破坏已清楚可见的当前帧 token。

在未遮挡且几何置信度高的随机节点集合 `\mathcal U` 上，以很低权重使用：

\[
\mathcal L_{keep}
=\frac{1}{|\mathcal U|}
\sum_{j\in\mathcal U}
\left(1-\cos(\hat h_j,h_j^{clean})\right).
\]

初始权重很小；若消融表明它抑制最终 QA，则移除。

## 2.7 不作为训练损失的项目

### Non-edge isolation

从严格的 block-diagonal sparse graph 看，若 distractor 与目标 clip 没有边，目标节点输出在数学上天然不受 distractor 影响。因此 `L_iso` 会退化为零或近似零，**不能作为核心训练 loss**。

它保留为 Phase 1/2 的诊断：检查实现是否意外存在跨 clip attention、batch normalization 泄漏或错误图边。

### Permutation consistency

对纯 sparse transport，图和节点同步重排时本来就具有置换等变性。它不作为 TIP 主损失；仅在 Phase 2 的静态多图 QA 子集做输入增强与答案一致性检查。

## 2.8 Phase 1 总损失

\[
\boxed{
\mathcal L_{TIP}
=
\mathcal L_{rec}
+
\lambda_{sub}\mathcal L_{sub}
+
\lambda_{keep}\mathcal L_{keep}
}
\]

初始搜索范围：

```yaml
tip:
  tau_nce: 0.07
  lambda_sub: [0.10, 0.25, 0.50]
  lambda_keep: [0.00, 0.02, 0.05]
  mask_ratio: [0.10, 0.15, 0.20]
  min_distinct_source_frames_for_substitution: 2
  source_edge_dropout: [0.0, 0.15]
```

## 2.9 Phase 1 训练配置

```yaml
model:
  qwen_checkpoint: Qwen/Qwen3-VL-4B-Instruct
  freeze_vision: true
  geowire_blocks: 2
  top_k_edges: 4
train:
  precision: bf16
  global_batch_size: 96
  lr_geowire: 2.0e-4
  weight_decay: 0.05
  warmup_ratio: 0.05
  max_steps: 30000
  optimizer: AdamW
```

每 1k step 评估：

```text
full graph vs self-loop only
full graph vs random graph (same degree)
full graph vs shuffled endpoints (same weights)
track-only vs projective-only vs union
masked recovery Recall@1/5 and cosine
support substitution consistency
non-edge isolation diagnostic
```

## 2.10 Phase 1 停止条件

未满足以下全部条件前，不进入 Phase 2：

1. full graph 在 masked recovery 与 Recall@K 上稳定胜过 random/shuffled graph；
2. union graph 不低于最优单边源；
3. direct-observation preservation 没有明显恶化；
4. non-edge isolation 诊断无异常泄漏；
5. 关键失败案例可被图质量或可见性解释，而不是 token layout bug。

---

# 3. Phase 2：Spatial Instruction Tuning

## 3.1 目的

将已经验证的 geometry-wired semantic transport 接入 Qwen merger 与 language decoder，验证其是否提高真实多帧/多视图空间问答能力。

## 3.2 可训练参数

| 组件 | 状态 |
|---|---:|
| VGGT-1B | 冻结，离线缓存 |
| Qwen3-VL visual encoder | 冻结 |
| GeoWire | 继续训练 |
| Qwen visual merger | LoRA / 小范围训练 |
| Qwen language decoder | LoRA |

默认不对视觉编码器做 LoRA。只有主结果已经确认、且严格受控实验显示有额外收益时，才做视觉 encoder LoRA 的补充消融。

## 3.3 训练数据：Core-Clean 主协议

核心原则：**不能用含有评测场景、评测自动生成 QA 或同一视频不同 clip 的训练数据换取分数。**

### 数据混合

| 数据 | 初始比例 | 使用内容 | 作用 |
|---|---:|---|---|
| SPAR-7M-RGBD | 50% | 仅 train、multi-view/video；PosMatch、ViewChg、CamMotion、ObjRel_MV、SpImag_MV 为主 | 多视图关系、参考系、组合空间推理 |
| XVR train | 30% | Correspondence、Verification、Localization | 直接训练跨视图对应、验证与定位 |
| LLaVA-Hound / LLaVA-Video | 20% | 连续片段中可回答的 temporal/spatial QA；非空间问题低采样 | 保留自然视频语义、遮挡、动作与时间语言 |

建议初始规模：`220k–300k QA`，每个样本默认 8 帧。SPAR 的绝对 metric depth/distance QA 最多占 SPAR 子集的 10%，因为 GeoWire 第一篇主张是 correspondence-based relation reasoning，不是 metric calibration。

### 严禁直接加入主训练的来源

- `VLM-3R-DATA`：其官方仓库明确说明训练集包括 VSiBench / VSTiBench instruction data；用于 VSI 主表会造成不清晰甚至直接的数据污染。
- `VSI-Bench`、`MMSI-Bench`、`ViewSpatial-Bench`、`DSR-Bench` 的 test/validation QA 或同场景衍生 QA。
- `VSI-590K`：在未完成 source scene / video ID 审计前禁止加入 Core-Clean；通过审计后只在单独标记的 **Scaled** regime 使用。

### 场景级去重

- VSI-Bench 来自 ScanNet、ScanNet++、ARKitScenes 的 288 个 validation videos；SPAR 包含 ScanNet 与 ScanNet++ 数据，必须按 `scene_id` 排除与评测重合的场景。
- 对 XVR、LLaVA-Hound 与后续数据也建立 source manifest；只要能映射到评测 scene/video，均排除。
- 对所有训练 run 保存 `leakage_audit.json`。

## 3.4 Scaled regime（补充，不替代 Core-Clean）

只有 Core-Clean 机制成立后，才做一个单独标记的扩展模型：

```text
Core-Clean mixture
+ scene-audited VSI-590K
+ 可审计的 Cambrian-S / PhysGame spatial subset
```

该模型可用于追赶依赖大规模 spatial SFT 的同期方法，但论文必须将它与 Core-Clean 分表，不可把其结果伪装成同等训练条件。

## 3.5 Phase 2 任务

1. **Spatial QA generation**：标准自回归回答，覆盖跨视图对应、物体关系、相机关系、视角变化、遮挡、时间变化、多步关系。
2. **Interleaved TIP retention**：每 3 个 QA batch 后插入 1 个 Phase 1 TIP batch，防止 SFT 将 GeoWire 退化为普通 residual adapter。
3. **Static view-order augmentation（仅静态任务）**：对 SPAR/XVR 中不依赖时间方向的问题重排输入图像，答案不变；这是数据增强与评测检查，不构成新的损失模块。

## 3.6 Phase 2 损失

QA batch：

\[
\mathcal L_{QA}
=-\sum_m\log p_\theta(y_m\mid y_{<m},q,H,P_G).
\]

TIP batch：

\[
\mathcal L_{TIP}
=
\mathcal L_{rec}+
\lambda_{sub}\mathcal L_{sub}+
\lambda_{keep}\mathcal L_{keep}.
\]

优化目标为交替 batch 的期望：

\[
\boxed{
\mathcal L_{SIT}
=
\mathbb E_{B_{QA}}[\mathcal L_{QA}]
+
\lambda_{TIP}\mathbb E_{B_{TIP}}[\mathcal L_{TIP}]
}
\]

实现上采用 `3 QA : 1 TIP` batch schedule；不要将被掩码的 TIP 视觉输入与 QA label 强行混在同一个样本中。

初始参数：

```yaml
phase2:
  qa_to_tip_batch_ratio: 3:1
  lambda_tip_effective: 0.20
  lr_geowire: 1.0e-4
  lr_lora: 2.0e-5
  warmup_ratio: 0.03
  max_steps: 12000
  lora_rank: 32
```

## 3.7 Phase 2 评测

主表：

```text
VSI-Bench + VSI-Bench-Debiased
MMSI-Bench
ViewSpatial-Bench
```

机制分析：

```text
masked recovery
full/random/shuffled graph gap
track-only / projective-only / union
occlusion-heavy and cross-view subsets
frame count 4 / 8 / 16 / 32
```

只有原始 VSI 上升、而 Debiased、MMSI 或机制诊断不提升时，不得宣称 GeoWire 改善了几何推理。

---

# 4. Phase 3：可选扩展，不属于主模型

触发条件：Phase 1 和 Core-Clean Phase 2 都已通过。

允许尝试：

1. `GeoVR-initialized Qwen3-VL + GeoWire`：验证节点级 geometry awareness 与边级 topology transport 的正交性；
2. 16/32 帧训练与 DSR-Bench；
3. scene-audited VSI-590K scaling；
4. StreamVGGT / VGGT-Ω 的长视频替换；
5. 视觉 encoder LoRA 的补充消融。

不允许将 Phase 3 的增强写成 GeoWire 主方法必需组成。

---

# 5. 最终数据—任务映射

| 阶段 | 数据 | 是否需 QA | 学到什么 | 主要输出 |
|---|---|---:|---|---|
| Phase 0 | SPAR-RGBD + LLaVA-Hound dev clips | 否 | 对齐与图认证 | `P_G` cache |
| Phase 1 | SPAR-RGBD + LLaVA-Hound 质量筛选 clips | 否 | 正确边上的 token 恢复与可替代性 | GeoWire adapter |
| Phase 2 | SPAR-MV/Video + XVR + LLaVA-Hound spatial QA | 是 | 将受约束 transport 转化为空间问答 | GeoWire + LoRA adapters |
| Phase 3 | audited VSI-590K / GeoVR / long-video data | 可选 | 规模化、正交增强、动态泛化 | supplementary checkpoints |

---

# 6. 论文中应如何描述训练

> We first pretrain the transport adapter without language supervision. Given frozen semantic patch tokens and a frozen VGGT-derived correspondence graph, we mask a target observation and require GeoWire to recover its semantic representation solely from geometrically certified observations in other frames. We further enforce that distinct valid views of the same local 3D support are interchangeable. We then instruction-tune the language-side modules for spatial QA while interleaving the transport pretraining objective, preserving the geometry-governed communication behavior during answer supervision.

中文：

> 我们首先在无需语言标注的条件下预训练 transport adapter。给定冻结的语义 patch token 与由 VGGT 构建的 correspondence graph，模型需要在当前观察被遮挡时，仅依赖其他帧中经过几何认证的观测恢复该位置的语义表征；同时，不同有效视角提供的对应证据应具有可替代性。随后，模型在空间问答上进行指令微调，并交替保留 transport 预训练目标，从而避免答案监督将几何控制的通信机制退化为普通视觉残差适配器。

---

# 7. References for data and protocol

- SPAR official repository: multi-view/video spatial QA, RGBD release with depth, pose and intrinsics, and source-scene organization.
- XVR official project: 100K multi-view VQA examples for correspondence, verification and localization.
- VG-LLM official repository: public `spar_234k` and LLaVA-Hound data preparation protocol for spatial reasoning.
- GeoThinker official repository: spatial instruction data usage and explicit treatment of VLM-3R / VSI-590K sources.
- VLM-3R official repository: states that its instruction data include VSiBench/VSTiBench training data.
- VSI-Bench official repository: test QA originate from 288 validation videos of ScanNet, ScanNet++ and ARKitScenes.
