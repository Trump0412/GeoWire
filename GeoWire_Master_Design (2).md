# GeoWire：Geometry-Wired Cross-Frame Semantic Transport for Video Spatial Reasoning

> **项目顶层设计文档 / Single Source of Truth**  
> 用途：统一约束后续代码实现、训练实验、结果记录、AAAI 论文写作与答辩叙事。  
> 状态：`v0.2 — architecture frozen, training protocol frozen`  
> 最后更新：2026-07-05

> **v0.2 training amendment**：训练阶段、数据混合与 TIP 损失以 `GeoWire_Training_Protocol.md` 为准；其明确覆盖旧版 `L_iso`、未遮挡 substitution 与未审计训练数据的描述。

---

## 0. 文档治理规则

本文件是 GeoWire 项目的最高层设计约束。之后的所有实验、配置、日志、图表、论文文字和 Codex 实现计划都应以本文件为准。

### 0.1 可修改与不可修改内容

| 类别 | 当前状态 | 修改规则 |
|---|---|---|
| 核心问题定义 | 冻结 | 只有在 Phase 1 失败后才允许改变 |
| GeoWire 主架构 | 冻结 | 不新增 bank、router、实体记忆或几何 feature fusion |
| 几何模型 | 冻结为 VGGT | 第一篇不切换至 DA3；可做 VGGT-Ω 的附录验证 |
| 语言模型主干 | 正式首跑暂定 Qwen3-VL-2B | 可替换同族 checkpoint，但不得改变“visual encoder → GeoWire → merger → LLM”的接口；Qwen3-VL-4B 保留为 pilot / comparison |
| 训练损失、边阈值、采样策略 | 开放 | 通过实验和消融更新 |
| benchmark 优先级 | 主表冻结 | 新 benchmark 只能作为补充，不可挤占主表 |
| 论文创新主张 | 冻结 | 只围绕 control-plane 与 topology intervention，不宣称新任务 |

### 0.2 任何新设计必须回答的四个问题

新增模块、损失或数据前，必须在本文件的变更记录中回答：

1. 它是否直接增强“几何决定语义通信拓扑”这一主线？
2. 它是否可以由现有 self-loop、稀疏边置信度或两层 transport 代替？
3. 它是否将模型重新带回 geometry feature fusion、query routing 或 object memory？
4. 它是否有可分离的消融和可证伪假设？

任一问题的答案不清楚，则不进入主模型。

---

# 1. 项目身份与一句话命题

## 1.1 论文题目（工作名）

**GeoWire: Geometry-Wired Cross-Frame Semantic Transport for Video Spatial Reasoning**

中文：**GeoWire：面向视频空间推理的几何驱动跨帧语义传输**

## 1.2 一句话命题

> **几何不是供语言模型读取的附加内容；几何是规定语义证据能够跨越哪些帧与位置传播的控制平面。**

英文版本：

> **Geometry is not auxiliary content to be fused into an MLLM; it is a control plane that determines where semantic evidence may propagate across frames.**

## 1.3 研究对象

GeoWire 研究的是已有问题：**多帧视频 / 多视图图像条件下的空间理解与推理**。

输入是一组图像帧和问题；输出是空间关系、参考系关系、遮挡、时空变化、路径或多图推理答案。项目不提出一个新的主任务，也不将自建诊断集包装为新 benchmark。

---

# 2. 问题、缺口与研究假设

## 2.1 问题

现有多帧 VLM 具有较强的二维语义能力，但在跨帧空间推理中经常出现两类结构性失败：

1. **同一局部三维区域的语义证据无法稳定汇聚。**  
   一个目标在第 1 帧清楚、在第 8 帧模糊或遮挡；模型缺少可靠机制把第 1 帧的语义证据迁移到第 8 帧。

2. **外观相似但空间不同的区域被错误混合。**  
   多把椅子、多个门、重复纹理和相似背景会让全局 cross-frame attention 误把无关 patch 当作同一来源。

当前方法多将 pointmap、depth、camera 或 VGGT feature 作为额外 token、key/value 或融合 feature 注入 VLM。这样增加了几何内容，却没有明确规定跨帧语义信息应沿哪些真实对应关系通信。

## 2.2 核心研究假设

**H1 — topology hypothesis**  
若使用 VGGT 的 track、visibility、pointmap、camera 和 depth 构建稀疏 correspondence graph，并只允许语义 patch token 沿该图跨帧传播，则可减少相似外观干扰并增强遮挡、视角变化和跨帧关系推理。

**H2 — intervention hypothesis**  
若训练时通过有对应支持的局部遮挡、等价视角替换和几何不连通干扰，迫使模型只依赖正确 correspondence 恢复证据，则其在 debiased spatial benchmarks 上的提升应高于普通 QA SFT。

**H3 — orthogonality hypothesis**  
GeoVR 类型的节点级几何蒸馏增强单个 token 的几何内涵；GeoWire 约束 token 之间的跨帧通信。两者若都有效，应表现为正交增益，而非彼此替代。

## 2.3 可证伪条件

若出现下列任一情况，则不得继续扩展复杂设计：

- GeoWire 在 correspondence-masked semantic recovery 上不优于无图结构或随机图；
- 在非边干扰测试中，GeoWire 不能降低错误证据泄漏；
- 在 VSI-Bench-Debiased、MMSI 或 ViewSpatial 的对应子任务中没有稳定收益；
- 收益只来自更长输入、更多 token 或更大模型，而不是正确 graph transport。

---

# 3. 理论立场与方法定位

## 3.1 三种融合范式

### A. Geometry as content

\[
Z=\Phi(S,G)
\]

几何 feature 与语义 feature 被 add、concat 或投影后混合。VG-LLM 属于这一类：其公开实现以 VGGT 几何编码器和常规视觉编码器处理视频，再在 patch 级融合特征并输入 Qwen2.5-VL。 [R2]

### B. Semantics queries geometry

\[
Z=\Phi\big(S,\operatorname{Attn}(Q(S,q),K(G),V(G))\big)
\]

问题或语义 token 主动查询几何 token。GeoThinker、若干 geometry-aware cross-attention 方法可归于此类。

### C. GeoWire: geometry controls semantic topology

\[
\boxed{Z=\Phi(S,P_GS)}
\]

其中：

- \(S\)：VLM 视觉编码器输出的语义 patch tokens；
- \(G\)：VGGT 的相机、深度、pointmap、tracks、visibility、confidence；
- \(P_G\)：由几何结果构建的稀疏 correspondence operator；
- 被跨帧传递的 value 始终是语义 token，而非 pointmap、depth 或 VGGT hidden features。

GeoWire 的立场是：

> **Geometry controls admissibility; self-loops preserve direct observation; language reasons afterward.**

## 3.2 主张的严格边界

GeoWire 可以主张：

- 将几何作为语义通信的 **control plane**；
- 通过 correspondence graph 约束跨帧 semantic transport；
- 用 topology intervention 验证模型是否真正使用有效几何路径。

GeoWire 不应主张：

- 首次使用几何信息提升 VLM；
- 首次做跨帧注意力；
- 首次使用 VGGT；
- 首次做几何蒸馏；
- 提出新的空间推理任务。

---

# 4. 最终主架构（冻结）

```text
Multi-frame images / video clip
   │
   ├── Frozen VGGT-1B
   │      ├── camera intrinsics / extrinsics
   │      ├── depth + confidence
   │      ├── pointmap
   │      └── tracks + visibility + confidence
   │                    │
   │                    └── Sparse correspondence operator P_G
   │
   └── Qwen3-VL Vision Encoder
          └── semantic patch tokens H
                         │
                         ▼
        GeoWire Sparse Transport × 2
        H ← H + α · P_G H
                         │
                         ▼
              Original visual merger / projector
                         │
                         ▼
                 Qwen language decoder
                         │
                         ▼
                       Answer
```

VGGT is used only as a frozen geometry control-plane provider. Its official implementation predicts camera parameters, point maps, depth maps and 3D point tracks from one to hundreds of views. [R1]

## 4.1 模块职责

| 模块 | 输入 | 输出 | 是否训练 | 职责 |
|---|---|---|---:|---|
| VGGT-1B | 多帧 RGB | pose, depth, pointmap, track, visibility, confidence | 否 | 构建几何通信图 |
| Qwen3-VL visual encoder | 同一组 RGB | patch token \(H\) | 默认冻结 | 提供语义 payload |
| Graph builder | VGGT 输出 + patch 坐标 | 稀疏 \(P_G\) | 否 | 决定合法语义通信 |
| GeoWire ×2 | \(H,P_G\) | 更新后的 \(H'\) | 是 | 沿几何边传输语义 |
| Visual merger | \(H'\) | multimodal tokens | LoRA / 小范围训练 | 对齐至语言空间 |
| Qwen LLM | multimodal tokens + question | answer | LoRA | 问题理解、证据组合与生成 |

## 4.2 唯一插入位置

GeoWire 仅插入在：

\[
\text{Vision Encoder} \rightarrow \boxed{\text{GeoWire}} \rightarrow \text{Visual Merger}
\]

不插入原始像素层，不插入 LLM decoder，不跨层注入 VGGT features。

理由：视觉 token 此时已经拥有局部语义，同时仍保留与二维 patch grid 的显式对应；进入 visual merger 之后，patch 坐标结构通常已被压缩，无法可靠建立 correspondence transport。

## 4.3 明确禁止的组件

以下组件不属于主模型，除非本文件经过正式版本更新：

- entity / object memory；
- geometry bank 或 continuity bank；
- query router；
- self/local/continuity competition；
- geometry feature concat、add 或 cross-attention injection；
- VGGT multi-layer hidden feature fusion；
- LLM 内部 geometry injection；
- 3D box、instance segmentation 作为必需中间表示；
- 为了“第二创新”而新增的 token gate。

这些设计会使主线退化为 feature fusion、语义主导检索或对象级状态融合。

---

# 5. Geometry Control Plane：稀疏 correspondence operator

## 5.1 token 与坐标对齐

设输入有 \(F\) 帧，每帧 Qwen visual encoder 输出 \(P\) 个 token：

\[
H\in\mathbb R^{F\times P\times d}
\]

每个 token \(i=(t,p)\) 都有图像平面中心坐标 \((u_{t,p},v_{t,p})\)。必须实现精确的 preprocessing coordinate transform，使 Qwen patch center 与 VGGT 输入坐标严格对应。该变换是工程正确性的第一优先级。

## 5.2 两类几何边

\[
\mathcal E=\mathcal E_{\text{track}}\cup\mathcal E_{\text{proj}}\cup\mathcal E_{\text{self}}
\]

### Track edges

以 Qwen patch center 为 query point，调用或读取 VGGT track head 的跨帧位置、visibility 和 confidence。

若源 token \((t,p)\) 在目标帧 \(s\) 中对应位置为 \((\hat u,\hat v)\)，则映射至最近的 Qwen patch \((s,q)\)。只有满足可见性、置信度、画面范围约束时保留边。

适用：动态局部区域、相机运动、部分可见物体。

### Projective edges

对 pointmap 点 \(X_{t,p}\)，依据目标相机 \((K_s,T_s)\) 重投影：

\[
\tilde x_{s,p}=\pi(K_sT_sX_{t,p})
\]

若投影落在目标帧内，且预测深度与目标帧 depth 一致，则连接到对应 patch：

\[
|D_s(\tilde x_{s,p})-z_{s,p}|<\tau_d
\]

适用：静态背景、track 覆盖不足区域。

### Self-loop

每个 token 均保留自环，直接观察永远不会被跨帧 transport 覆盖。

## 5.3 边认证与权重

每条候选边必须经过以下几何检验：

- VGGT track confidence；
- visibility；
- pointmap 重投影误差；
- 深度一致性；
- \(t\rightarrow s\rightarrow t\) cycle consistency；
- 目标位置是否落在图像有效区域。

边权仅由几何量决定：

\[
 c_{ij}=c^{\text{track}}_{ij}\cdot v_{ij}\cdot
 \exp(-e^{\text{reproj}}_{ij}/\tau_r)\cdot
 \exp(-e^{\text{cycle}}_{ij}/\tau_c)
\]

每个 token 保留 top-\(K\) 个最高置信跨帧邻居，初始设定 \(K=4\)。加入 self-loop 后行归一化：

\[
P_G(i,j)=\frac{\mathbf1[(i,j)\in\mathcal E]c_{ij}}
{\sum_{k\in\mathcal N_G(i)}c_{ik}}
\]

### 不允许进入边权的量

- 文本问题 embedding；
- LLM hidden state；
- 语义相似度；
- 可学习 query router；
- token 内容 gate。

原因：这些量会使语义重新获得“创造或改变通信拓扑”的权力，破坏 control-plane 定义。

## 5.4 “bank” 的最终处理

不维护 semantic bank 或 geometry bank。

唯一允许缓存的是：

```text
per-clip sparse correspondence index
{src_token_idx, dst_token_idx, edge_weight, edge_type, metadata}
```

它是无参数 CSR / COO 图数据结构，用于 sparse gather/scatter；不是可学习 memory，不跨 clip 累积，不做 query 检索。

---

# 6. GeoWire Sparse Semantic Transport

## 6.1 基本更新

\[
H^{(0)}=H
\]

\[
H^{(\ell+1)}=H^{(\ell)}+
\alpha_\ell W_o\left(P_G\cdot\operatorname{LN}(H^{(\ell)})W_v\right),
\quad \ell\in\{0,1\}
\]

其中：

- \(P_G\) 为冻结且 detached 的稀疏几何算子；
- \(W_v,W_o\) 为 GeoWire 内的小型线性投影；
- \(\alpha_\ell\) 是每层一个可学习残差标量，初始化为 0；
- transport 的 value 永远来自语义 token \(H\)。

## 6.2 为什么使用两个 block

- Block 1：一跳 correspondence semantic transport；
- Block 2：有限两跳传播，覆盖“当前帧遮挡 → 中间帧部分可见 → 远帧清晰可见”的情况；
- 不使用三个以上 block，以避免语义扩散、过度平滑和错误边累积。

块数不是创新点，而是必须做的深度消融：\(0,1,2,3\) blocks。

## 6.3 门控原则

| 机制 | 是否使用 | 定义 |
|---|---:|---|
| 几何边置信度 | 是 | 构建 \(P_G\)，决定边是否允许存在 |
| 残差标量 \(\alpha_\ell\) | 是 | 稳定训练，逐渐启用 transport |
| self-loop | 是 | 保留当前帧直接观察 |
| token-level semantic gate | 否 | 会变成 router / competition |
| text-conditioned gate | 否 | 会让问题文本改变通信拓扑 |
| visual-vs-geometry gate | 否 | 会退化为 GeoSR 型 feature trust allocation |

---

# 7. 第二贡献：Topology Intervention Pretraining（TIP）

GeoWire 的第二贡献不是新 architecture，而是专门训练模型尊重并使用 correspondence topology 的机制。

## 7.1 目标

普通 QA SFT 无法证明模型真的沿几何边取证。TIP 构造三类 intervention，使“正确的 correspondence”成为唯一可靠的跨帧证据来源。

## 7.2 Intervention A：Correspondence-Masked Semantic Recovery

选取当前帧可见、且在其他帧存在高置信对应的 target token \(i\)。在 GeoWire 输入前遮掉该 token 的语义表示，要求模型经图传播恢复无遮挡语义。

\[
\mathcal L_{\text{rec}}=1-
\cos\left(\hat h_i,\operatorname{sg}(h_i^{\text{clean}})\right)
\]

teacher 是同一冻结 Qwen visual encoder 在无遮挡输入下的 token；不依赖额外人工标注。

## 7.3 Intervention B：Equivalent Support Substitution

同一局部三维位置可在多个帧中被可靠观察。随机替换可用 source correspondence 集合：

\[
\mathcal N_G^a(i)\leftrightarrow\mathcal N_G^b(i)
\]

要求 transport 后的 token 与最终答案保持稳定：

\[
\mathcal L_{\text{sub}}=
\left\|f(H,P_G^a)_i-f(H,P_G^b)_i\right\|_2^2
\]

它训练的不是“记住某一帧”，而是“同一三维局部的多个几何有效观察可互换”。

## 7.4 Intervention C：Non-edge Isolation

加入外观相似但与 target 不存在几何路径的 distractor frame / patch。由于其没有进入 \(P_G\)，目标 token 的表示与输出应保持稳定：

\[
\mathcal L_{\text{iso}}=
\left\|f(H,P_G)_i-f(H\cup H_{\text{dis}},P_G)_i\right\|_2^2
\]

该项直接测量模型是否抑制“相似但不对应”的错误信息泄漏。

## 7.5 可选 Intervention D：Permutation Consistency

只对静态多视图任务或不依赖时间顺序的问题启用。重排帧序与节点编号，同时相应重排图：

\[
\mathcal L_{\text{perm}}=
D_{\text{KL}}\big(p(y\mid H,P_G),p(y\mid \Pi H,\Pi P_G\Pi^\top)\big)
\]

不对时间先后、运动、路径类问题启用，以免破坏真实时序语义。

## 7.6 总训练目标

\[
\mathcal L=
\mathcal L_{\text{QA}}+
\lambda_1\mathcal L_{\text{rec}}+
\lambda_2\mathcal L_{\text{sub}}+
\lambda_3\mathcal L_{\text{iso}}+
\lambda_4\mathcal L_{\text{perm}}
\]

默认阶段性启用：

- Topology pretraining：\(\mathcal L_{\text{rec}}+\mathcal L_{\text{sub}}+\mathcal L_{\text{iso}}\)；
- Spatial SFT：\(\mathcal L_{\text{QA}}\) 与较低权重 TIP；
- Permutation consistency：仅静态多图训练子集。

---

# 8. 模型选择与资源策略

## 8.1 主模型

| 组件 | 选择 | 角色 |
|---|---|---|
| Geometry teacher | VGGT-1B | 冻结，离线几何 cache 与 graph builder |
| 主 VLM | Qwen3-VL-2B-Instruct | 正式首跑模型；优先缩短工程闭环并提高 batch ramp 余量，Qwen3-VL-4B-Instruct 保留为对照 |
| 小模型验证 | Qwen3-VL-2B-Instruct | 证明几何拓扑可在小模型上带来结构性收益 |
| 兼容性对照 | Qwen2.5-VL-7B-Instruct | 仅用于与 VG-LLM / 已有代码生态的公平比较 |

Qwen3-VL is an official open multimodal model family, while GeoVR reports geometry distillation experiments on small Qwen3-VL variants. [R5][R3]

## 8.2 训练范围

- VGGT：始终冻结；
- Qwen visual encoder：Phase 1 冻结；Phase 2 只在必要时 LoRA，不做全量解冻；
- GeoWire \(W_v,W_o,\alpha\)：全程训练；
- Visual merger / projector：LoRA 或小范围训练；
- LLM：LoRA，避免破坏原始语言与通用视觉能力。

## 8.3 计算资源

初始可用配置：6 × A100 80GB。

建议：

| 项目 | 初始值 |
|---|---|
| 训练帧数 | 8 帧 |
| 评测帧数 | 4 / 8 / 16 / 32 帧 |
| GeoWire blocks | 2 |
| 邻居数 \(K\) | 4，后做 2/4/8 消融 |
| 精度 | bf16 |
| 并行 | DeepSpeed ZeRO-2 + gradient checkpointing |
| VLM 微调 | QLoRA / LoRA rank 32 起步 |
| VGGT | 离线缓存，训练期不在线前向 |

---

# 9. 数据策略与泄漏控制

## 9.1 数据分层

### Phase 0：几何契约与缓存

使用有相邻帧、多视图或视频结构的数据，不追求语言规模。目标是验证：VGGT 输出与 Qwen patch grid 坐标对齐，边构建稳定，graph transport 可运行。

### Phase 1：Topology pretraining

优先使用具有多帧视觉证据的视频 / 多视图样本。候选来源：

- LLaVA-Hound video subset；
- SPAR 的多图 / 多视图空间样本；
- 具有公开相机或重建信息的室内视频；
- 合成或可控场景，仅用于构造 intervention，不作为主结果来源。

该阶段不依赖高质量语言答案，VGGT correspondence 自动生成 TIP supervision。

### Phase 2：Spatial instruction tuning

使用空间问答、视频空间推理和多图空间理解数据。候选包括 SPAR、空间 instruction 训练集及严格去重后的公开视频空间数据。

### Phase 3：可选规模化

只有 Phase 1/2 证实机制有效后，再考虑扩展数据和训练轮次。不得在未做机制验证前堆入全部 spatial QA 数据。

## 9.2 强制泄漏控制

主评测数据包含 ScanNet、ScanNet++、ARKitScenes 等来源时，训练集必须按 **scene identity** 去重，而非只按文件名或 QA ID 去重。

- VSI-Bench：来源为 288 个 egocentric indoor videos、超过 5,000 QA，来自公开室内重建数据验证集。 [R4]
- 不允许训练中出现同一 scene 的不同 clip、不同采样帧或自动生成 QA；
- 对公开训练数据的源数据集、split、scene ID 建立 manifest；
- 每次报告结果必须附上 `leakage_audit.json`。

---

# 10. Benchmark 与论文主表

## 10.1 主表：VSI-Bench 与 VSI-Bench-Debiased

用途：检验 video spatial intelligence、跨帧空间记忆、构型/测量/时空关系。

必须同时报告：

- 原始 VSI-Bench；
- VSI-Bench-Debiased；
- 4 / 8 / 16 / 32 帧输入；
- task category；
- uniform 与可复现 sampling protocol。

重点任务：遮挡恢复、跨帧关系、路线、空间变化、相似物体干扰。

## 10.2 主表：MMSI-Bench

用途：检验多图空间智能，尤其是多视角对应、camera-object、multi-step reasoning。MMSI-Bench 有 1,000 道专家设计的多图空间题，并报告当前开放模型与人类之间的大差距。 [R6]

重点报告：

- object-object；
- camera-object；
- camera motion；
- object motion；
- multi-step reasoning。

## 10.3 机制表：ViewSpatial-Bench

用途：检验相机视角与人 / 实体参考系的转换。该 benchmark 专门覆盖五类多视角空间定位任务。 [R7]

GeoWire 的预期优势：跨视角证据可由 correspondence 迁移；但它不是完整 metric geometry solver，因此绝对距离不是第一主张。

## 10.4 动态泛化表：DSR-Bench

用途：检验动态物体、可见性变化与时间顺序。该表在主模型机制确认后运行，不作为第一轮实现的前置条件。

## 10.5 诊断协议：GeoWire Diagnostics

不包装为新 benchmark。只作为机制分析：

| 指标 | 含义 |
|---|---|
| Masked Token Recovery | 有效对应被保留时，能否恢复被遮挡 token 的语义 |
| Transport Retrieval@K | transport source 是否命中独立对应真值或高质量 track |
| Equivalent-support consistency | 换一个有效 source view，输出是否稳定 |
| Non-edge leakage | 加入相似但图不连通的干扰后，target 表示/答案变化是否受控 |
| Edge ablation sensitivity | 打乱真实边、随机边、移除 visibility 后性能如何变化 |

---

# 11. 对比实验设计

## 11.1 外部已发表方法

| 标记 | 方法 | 角色 | 处理方式 |
|---|---|---|---|
| B0 | Qwen3-VL-2B / 4B | 无显式几何强基线 | 必须统一 sampler 与 evaluator 实跑 |
| B1 | VG-LLM | geometry-as-content / patch fusion | 优先用官方代码或 checkpoint；不足时做严格受控重实现 |
| B2 | GeoThinker | semantics-query-geometry / frame-strict 思路 | 使用官方协议或公开 checkpoint；注明 backbone / 数据差异 |
| B3 | GeoSR | geometry gate + anti-shortcut 训练 | 在可兼容 benchmark 复现或报告官方数值 |
| B4 | GeoVR | node-level geometry distillation | 结果对比；若代码或权重可用，做 node+edge 正交增强实验 |
| B5 | 其他同期空间 VLM | 参考上限 | 只有统一数据和 evaluator 时才并排比较 |

外部方法必须区分：**同协议实跑结果** 与 **论文报告结果**。不可将不同 backbone、不同帧数、不同训练数据和不同 evaluator 的数值直接并排得出结论。

## 11.2 受控内部对照

| 标记 | 变体 | 回答的问题 |
|---|---|---|
| C0 | Vanilla Qwen3-VL | 没有几何时的下限 |
| C1 | Raw VGGT feature fusion | 直接把几何当内容是否足够 |
| C2 | Frame-strict geometry use | 只允许同帧几何辅助是否足够 |
| C3 | Dense temporal attention + geometry bias | 仅增加相对几何偏置是否足够 |
| C4 | GeoWire, random graph | 是否确实需要正确 correspondence |
| C5 | GeoWire, shuffled graph | graph 拓扑是否是增益来源 |
| C6 | GeoWire, track edges only | 动态对应贡献 |
| C7 | GeoWire, projective edges only | 静态重投影对应贡献 |
| C8 | GeoWire full | 完整主模型 |

## 11.3 必需消融

| 维度 | 设置 |
|---|---|
| blocks | 0 / 1 / 2 / 3 |
| top-K edges | 2 / 4 / 8 |
| edge source | track / projection / union |
| edge certification | no confidence / +visibility / +depth / +cycle |
| training | QA only / +recovery / +substitution / +isolation / full TIP |
| model scale | Qwen3-VL-2B / 4B |
| frames | 4 / 8 / 16 / 32 |

---

# 12. 实施路径与阶段门槛

## Phase 0：Geometry Contract

**目标**：验证几何图构建正确，不训练 VLM。

必须实现：

1. `Qwen patch center ↔ VGGT pixel coordinate` 双向映射；
2. track edge 可视化；
3. projective edge 可视化；
4. depth / reprojection / cycle consistency 检查；
5. 边覆盖率、边长度、visibility、edge type 分布统计；
6. 图像 overlay：source patch、target patch、真实/预测对应。

**门槛**：随机抽取 100 个 clip 人工检查；若边明显落在错误物体或错位 patch，禁止进入 Phase 1。

## Phase 1：Topology Pretraining

**目标**：证明 GeoWire 可以沿正确边恢复并隔离语义。

训练：GeoWire only；Qwen visual encoder 冻结；不需要 QA SFT。

关键指标：

- Masked Token Recovery；
- Random / shuffled graph 对照；
- Non-edge leakage；
- edge certification 消融。

**成功门槛**：完整图应明显优于随机图、打乱图和无图；否则项目问题在 graph 或 token alignment，不能继续给 LLM 加 LoRA。

## Phase 2：Spatial Instruction Tuning

**目标**：验证 topology transport 转化为空间 QA 分数。

训练：GeoWire + merger LoRA + LLM LoRA；保持 VGGT 冻结。

主评测：VSI-Debiased、MMSI-Bench、ViewSpatial-Bench。

**成功门槛**：相对于同 backbone 的 C0、C1、C3，在对应类任务取得一致收益；若仅原始 VSI 涨分而 debiased 与诊断不涨分，不得宣称几何推理提升。

## Phase 3：正交增强与扩展

仅在主线成立后执行：

- GeoVR-initialized backbone + GeoWire；
- 16/32 帧；
- DSR-Bench；
- causal / long-video extension；
- VGGT-Ω 或 StreamVGGT 替换实验。

这些属于扩展，不改变论文主方法。

---

# 13. 代码仓库蓝图

```text
geowire/
├── configs/
│   ├── phase0_graph.yaml
│   ├── phase1_tip.yaml
│   ├── phase2_sft.yaml
│   └── eval.yaml
├── data/
│   ├── clip_dataset.py
│   ├── spatial_sft_dataset.py
│   ├── intervention_dataset.py
│   └── leakage_manifest.py
├── geometry/
│   ├── vggt_cache.py
│   ├── coordinate_align.py
│   ├── graph_builder.py
│   ├── track_edges.py
│   ├── projective_edges.py
│   ├── certify_edges.py
│   └── graph_io.py
├── models/
│   ├── qwen_wrapper.py
│   ├── geowire.py
│   ├── sparse_transport.py
│   └── lora_targets.py
├── training/
│   ├── pretrain_tip.py
│   ├── train_sft.py
│   ├── losses.py
│   └── scheduler.py
├── evaluation/
│   ├── eval_vsi.py
│   ├── eval_mmsi.py
│   ├── eval_viewspatial.py
│   ├── diagnostics.py
│   └── compare_protocols.py
├── visualization/
│   ├── plot_edges.py
│   ├── plot_transport.py
│   ├── plot_interventions.py
│   └── render_case_study.py
├── tests/
│   ├── test_coordinate_alignment.py
│   ├── test_reprojection.py
│   ├── test_cycle_consistency.py
│   ├── test_sparse_transport.py
│   └── test_permutation.py
└── docs/
    ├── MASTER_DESIGN.md
    ├── EXPERIMENT_LOG.md
    └── PAPER_EVIDENCE.md
```

## 13.1 不可跳过的单元测试

- 坐标变换的 resize/crop/pad 与 VGGT/Qwen 预处理一致；
- identity camera + planar depth 的解析重投影；
- edge cycle 通过 / 失败样例；
- sparse scatter 与稠密矩阵实现数值一致；
- self-loop 时 GeoWire 可退化为原始视觉 token；
- \(\alpha=0\) 时整网输出与未插入 GeoWire 时一致；
- 静态多视图置换后，图与 token 索引同步重排。

---

# 14. 论文叙事与写作骨架

## 14.1 Introduction 的逻辑链

1. 多帧 MLLM 的失败并不只是“缺少更多几何 feature”，而是 semantic evidence 跨帧流动没有物理约束；
2. 前馈几何模型已能恢复 tracks、pointmaps、cameras、visibility 等 correspondence；
3. 现有方法将几何作为 content 融合、作为 attention target，或以 gate 分配 feature trust；
4. GeoWire 将几何提升为 control plane，构建稀疏 correspondence operator，约束 semantic token 的跨帧 transport；
5. TIP 以干预式训练排除 appearance / language shortcuts；
6. 在复杂 video/multi-image spatial benchmarks 与机制诊断上验证。

## 14.2 三项贡献（候选最终版本）

1. **Geometry-Wired Semantic Transport**：首次在本项目中将冻结前馈几何输出转为稀疏 control-plane operator，约束多帧 VLM patch token 的合法跨帧通信，而非融合几何 feature。
2. **Topology Intervention Pretraining**：提出有 correspondence 支持的遮挡恢复、等价证据替换与非边隔离训练，使模型可被验证地依赖正确几何拓扑。
3. **Comprehensive Analysis**：在 video、多图和多参考系 benchmark 上，结合图扰动、遮挡和相似物体干扰诊断，说明空间提升来自 correspondence-guided evidence transport。

注意：若文献核查发现“首次”表述过强，应写为“a geometry-as-control-plane formulation”，不使用绝对优先权措辞。

## 14.3 论文图表清单

| 图表 | 要回答的问题 |
|---|---|
| Fig. 1 Method overview | 几何 control plane 与语义 data plane 的分工 |
| Fig. 2 Graph construction | track / projection / certification / self-loop |
| Fig. 3 TIP interventions | recovery、substitution、isolation |
| Fig. 4 Qualitative transport | 遮挡目标从正确帧读取证据，避开相似干扰 |
| Table 1 Main benchmark | VSI-Debiased、MMSI、ViewSpatial |
| Table 2 External comparison | 与 feature fusion、frame-strict、geometry distillation 对比 |
| Table 3 Controlled ablation | random/shuffled graph、edge sources、blocks、TIP |
| Table 4 Robustness diagnostics | recovery、leakage、substitution consistency |
| Fig. 5 Frame scaling | 4/8/16/32 帧下性能与成本 |

---

# 15. 风险清单与应对

| 风险 | 表现 | 优先处理 |
|---|---|---|
| Qwen/VGGT patch coordinate mismatch | 边看似存在但对应到错误 patch | Phase 0 可视化与解析单元测试 |
| VGGT track 在动态区域不稳定 | 错边导致语义污染 | visibility/confidence/cycle 认证，宁可稀疏 |
| 过度传输导致语义过平滑 | 多帧越多反而下降 | 两层上限、残差初始化 0、K 小 |
| 模型依赖语言捷径 | 原 VSI 涨、debiased 不涨 | TIP、debiased 与干预诊断 |
| 模型依赖更多帧而非正确边 | random graph 仍涨 | shuffled/random graph 强对照 |
| 训练数据泄漏 | VSI/MMSI 分数异常高 | scene-level manifest 与审计 |
| Qwen3-VL 训练接口复杂 | 难以截取视觉 token 或插入模块 | 先完成 Qwen2.5-VL compatibility prototype，再迁移主模型 |

---

# 16. 与近期工作的关系

## 16.1 必须学习，但不照搬

| 工作 | 可借鉴 | 不应复制 | GeoWire 的关系 |
|---|---|---|---|
| VGGT | 相机/pointmap/track/visibility 的冻结几何供给 | 重新训练几何 backbone | control-plane provider |
| VG-LLM | 双编码器训练链路与 benchmark protocol | patch-level feature fusion | 主要 content-fusion 对照 |
| GeoThinker | 几何和语义错位会破坏融合的诊断 | query 主导几何读取 | GeoWire 将约束推进到跨帧通信图 |
| GeoSR | 打掉二维 shortcut 的训练动机 | feature gate / geometry injection | TIP 改为 topology intervention |
| GeoVR | 小模型几何激活、VGGT teacher 的节点级蒸馏 | camera/depth/scale multi-head 作为主线 | 节点级表征与边级通信的正交实验 |
| π³ | 避免固定参考帧偏置 | 修改几何 backbone | 静态多图的 permutation regularizer |
| SpatialStack / GeoAlign | 层位消融与几何细节可能早期丢失的观察 | 多层 feature injection | 只借实验诊断，不借主架构 |

## 16.2 GeoVR 的确切定位

GeoVR 的核心是以相机、深度、尺度和多尺度几何特征蒸馏，将几何知识内化到小型 Qwen3-VL 的表示空间。 [R3]

GeoWire 与之正交：

\[
\text{GeoVR}: \text{node geometry awareness}
\]

\[
\text{GeoWire}: \text{edge-valid semantic communication}
\]

因此，GeoVR + GeoWire 是后续的增强实验，而非核心依赖。主论文必须先证明 vanilla Qwen3-VL + GeoWire 的增益。

---

# 17. 当前冻结决策与开放问题

## 17.1 已冻结

- 使用 VGGT，不使用 DA3；
- 主线是 geometry-as-control-plane；
- 不进行实体化、object memory 或 entity bank；
- 只对 Qwen semantic tokens 做 transport；
- 不向 VLM/LLM 注入 VGGT hidden features、depth/pointmap tokens；
- 使用 track edge + projective edge + self-loop；
- 只在 vision encoder 与 visual merger 之间插入两层 GeoWire；
- 训练以 TIP + spatial SFT 两阶段进行；
- 主评测为 VSI-Debiased + MMSI，ViewSpatial 为机制表。

## 17.2 开放但有边界的问题

| 问题 | 可选范围 | 决策依据 |
|---|---|---|
| 主 VLM | Qwen3-VL-2B 正式首跑；Qwen3-VL-4B 与 Qwen2.5-VL-7B 对照 | 代码可插入性、稳定性、可比性 |
| graph 采样 | patch center / 多点采样 / adaptive sampling | Phase 0 edge precision 与覆盖率 |
| edge 数 K | 2/4/8 | 性能、泄漏、算力 |
| block 数 | 1/2/3 | 过平滑与任务收益 |
| tip 权重 | grid search | debiased 性能与诊断指标 |
| training clips | 4/8/16 | memory、correspondence coverage 与收益 |
| GeoVR 初始化 | 可选 | 是否带来正交增益 |

---

# 18. 变更记录

| 版本 | 日期 | 内容 |
|---|---|---|
| v0.1 | 2026-07-05 | 确立 GeoWire control-plane 叙事；冻结 VGGT → sparse graph → semantic transport → Qwen merger/LLM 架构；明确删除 bank、router、competition 和多层 geometry fusion；定义 TIP、主 benchmark、代码蓝图与实验门槛。 |

---

# 19. 参考来源（维护用）

[R1] Wang et al. *VGGT: Visual Geometry Grounded Transformer*. CVPR 2025. Official repository: facebookresearch/vggt.  
[R2] Zheng et al. *Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors* (VG-LLM). 2025. Official repository: LaVi-Lab/VG-LLM.  
[R3] Wang and Huang. *Learning Geometric Representations from Videos for Spatial Intelligent Multimodal Large Language Models* (GeoVR). arXiv:2606.05833, 2026.  
[R4] Chen et al. *Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces* (VSI-Bench). 2024. Official repository: vision-x-nyu/thinking-in-space.  
[R5] Qwen Team. *Qwen3-VL*. Official repository: QwenLM/Qwen3-VL.  
[R6] Xu et al. *MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence*. ICLR 2026. Official repository: InternRobotics/MMSI-Bench.  
[R7] Li et al. *ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in Vision-Language Models*. 2025. Official project and repository: ZJU-REAL/ViewSpatial-Bench.  
[R8] GeoThinker, GeoSR, SpatialStack, π³, DSR Suite and other related work must be added with final bibliographic metadata before paper submission. Their roles are fixed in Section 16; their exact citation details remain to be verified during related-work writing.

---

## Appendix A. 最终自检句

在任何设计会议、代码 review 或论文修改前，用下面一句话检验是否偏离主线：

> **Does this change make geometry decide where semantic evidence can safely travel, or does it merely inject more geometry features into the model?**

若答案是后者，该改动不属于 GeoWire 主模型。


---

# v0.2 训练协议覆盖说明

`GeoWire_Training_Protocol.md` 是本文件关于训练流程、损失函数、数据混合与泄漏控制的正式补充。以下旧设计已废止：

- 将 `Non-edge Isolation` 作为可优化损失：严格不连通图下该损失在结构上平凡，应只作为诊断；
- 在 target 未遮挡时执行 `Equivalent Support Substitution`：会被 self-loop 绕过；替代一致性必须与 target masking 联合执行；
- 将 `Permutation Consistency` 作为 TIP 主损失：它主要是图的结构性质，只在静态多视图 QA 中作为数据增强和评测检查；
- 未经 source scene/video 审计直接混入 `VSI-590K` 或 `VLM-3R-DATA`。

最终训练阶段为：

```text
Phase 0: geometry contract / cache, no optimization
Phase 1: TIP, train GeoWire only
Phase 2: spatial instruction tuning, train GeoWire + merger LoRA + LLM LoRA
Phase 3: optional scaling/orthogonal extension
```
