# 基于 ABot-Recon 的流式锚帧选择

对单目 RGB 视频逐帧输出 ABot 的三维预测。选择器观察若干张已经到达的后续帧，再决定是否把候选帧存为长期锚。每次选择均记录候选帧、决定发生时的输入帧、参考锚、覆盖率、视差和选择原因。候选与锚帧的序号从 0 开始。

## 运行

使用已安装官方 ABot-Recon、PyTorch、Pillow、NumPy 和 OpenCV 的 Python 环境。命令行使用 CUDA 和 SDPA。输入可以是按文件名排序的 RGB 目录，也可以是 TUM 的 `rgb.txt`，后者每行给出时间戳和相对路径。可指定只含所需采样帧的清单。输出目录须为新目录。[ABot-Recon 安装与权重](https://github.com/amap-cvlab/ABot-Recon)

```bash
python -m stream_reconstruction rgb.txt --checkpoint /path/to/abot_recon.safetensors --output run-01
python -m unittest discover -s tests -v
```

`decisions.jsonl` 中每个输入帧均有一条决定。Python 入口为 `stream_reconstruction.abot.reconstruct(network, paths, config)`。每条流独立保存视觉历史和选帧状态。流结束后还会发送尾帧的决定，此时 `prediction` 为空。

## 主干的 24 个视觉位置

默认 `--visual-context structured24`。每个跨帧注意力层用当前帧的 Query 读取以下视觉 K/V，然后才预测相机位姿、局部点图和置信度。每个位置都含一整帧的寄存器和空间 patch。每个输入步只编码一张新图。

| 位置 | 当前第 `t` 帧能读取的内容（从 0 编号） |
| --- | --- |
| 12 个近期位置 | 当前帧及之前最多 11 帧。 |
| 6 个历史锚位置 | 基于下文的共视和视差判据接纳帧。离开近窗且已确认后进入有界锚库，从库中按当前图像描述子的余弦相似度取前 6 张。 |
| 3 个固定位置 | 帧 0、1、2 各自离开近窗后使用独立位置。固定帧与历史锚按帧号去重。 |
| 3 个全历史摘要位置 | 所有离开近窗的帧各贡献一次。最早三帧各建一个簇，接着三帧按相似度为每簇再分配一个成员。随后按与簇平均描述子的余弦相似度归类。每簇逐层、逐 patch 以 FP32 更新 K/V 均值，再以推理精度提供给注意力。 |

真实帧使用原始帧号的三维 RoPE。每个簇将键的时间通道转到首个成员的相位后求均值，空间编码沿用原 patch 网格。摘要按图像网格聚合，尚未做跨视角几何对齐。真实帧可以同时为聚合摘要贡献信息。若摘要只有一个成员且该帧已直接入窗，则跳过该摘要。短序列和锚帧不足时按实际可用数运行。

`trajectory.jsonl` 的 `visual_context` 记录近期帧号、窗外锚、固定帧、摘要支持数与起止帧号。`positions` 为本步位置数，`key_tokens` 从传给注意力的实际键张量读取。发布配置在 504×280 分辨率下每个位置有 725 个 token，满额时每层实际读取 17,400 个。用同一输入、同一权重对照两种视觉窗口可执行以下命令。

```bash
python -m stream_reconstruction "$RGB" --checkpoint "$CHECKPOINT" --output runs/local12 --visual-context local12 --pose-mode baseline --profile
python -m stream_reconstruction "$RGB" --checkpoint "$CHECKPOINT" --output runs/structured24 --visual-context structured24 --pose-mode baseline --profile
```

下文的历史实验数字均来自 `local12`。几何求解的 `--summary` 和 `--ttt` 仍由各自开关控制。本实现对接了上游单帧解码器的缓存接口，使用 PyTorch SDPA 执行注意力。[上游核对版本](https://github.com/amap-cvlab/ABot-Recon/tree/6e3b2e1257464a74dfceff22eb33190bd228276e)

### 发布权重下的视觉窗口对照

在 RTX 3090 上以 BF16 和相同 RGB 输入对照两种窗口。两组都使用 `--pose-mode baseline`，轨迹评分直接使用模型的原生相机位姿。ATE 为整段 Sim(3) 对齐后的平移 RMSE。

| ScanNet 场景 | `local12` ATE（厘米） | `structured24` ATE（厘米） | 结构化窗口满额帧数 | 端到端 P95，12／24（毫秒） |
| --- | ---: | ---: | ---: | ---: |
| scene0070_00，1000 帧 | 26.69 | 97.58 | 921 | 146.8／137.6 |
| scene0000_00，5000 帧 | 40.56 | 174.19 | 4936 | 141.2／128.9 |

在同一帧与同一前缀状态下，分别移除固定帧、历史锚和视觉摘要，这三种操作都改变了正式权重输出的位姿、局部点图和置信度。400 帧消融在结构化路径上只保留近期帧时复现了原版的 11.74 厘米 ATE。分别加回固定帧、历史锚或摘要后，ATE 分别为 19.69、17.41 和 13.04 厘米。当前 `structured24` 用于研究实验，两个实测场景的轨迹精度均有退化。

### 视觉历史门微调

训练冻结 ABot 权重，只更新最后一层跨帧注意力中的六个系数。近期窗口提供注意力基准，固定帧、历史锚和摘要各自形成一项带权残差。训练复用实际的缓存与检索，用跨 1、8 和 16 帧的相对位姿监督编码后的当前预测。同一场景中的旧帧状态在反传时视为常量。

训练清单为 JSON，`train` 数组中每个场景应提供 `scene`、RGB 图像目录 `rgb`、连续前缀长度 `frames`、含 `poses` 的 `reference` NPZ 和 `baseline` 轨迹 JSONL。文件路径以清单位置为根。基线需含最初 120 帧的原生 `raw` 位姿，供训练时拟合尺度。训练逐轮输出权重，独立验证场景用于选权重。

```bash
python -m stream_reconstruction.train_visual_gate train.json \
  --checkpoint "$CHECKPOINT" --output runs/visual-gate --epochs 2
python -m stream_reconstruction "$RGB" --checkpoint "$CHECKPOINT" \
  --output runs/visual-tuned --pose-mode baseline --visual-context structured24 \
  --visual-gate-weights "$GATE_WEIGHTS" --profile
```

`GATE_WEIGHTS` 指向选中的轮次 JSON。`trajectory.jsonl` 中的 `visual_context.history_weights` 依次记录固定帧、历史锚和摘要的融合权重。`key_tokens` 在视觉门运行中对应单次注意力调用的最大键数。

首轮实验在三个 ScanNet 场景训练两轮，以 `scene0097_00` 的 600 帧整段 ATE 选出第 2 轮。两条 1000 帧测试序列上，原始 12 帧、直接 24 位置、初始门和微调门的整段 ATE 在 `scene0140_00` 分别为 16.67、51.57、16.73、17.46 厘米，在 `scene0070_00` 分别为 26.69、97.58、26.78、22.27 厘米。微调门的 500 帧间隔旋转误差在两条序列均略高于原始 12 帧。

加上 `--long-horizon` 可在 1、8、16 帧之外监督 32 和 64 帧的相对位姿。默认每 8 帧更新一次梯度块，组合块内的相邻位姿，并在块边界截断梯度。旋转与平移分别计损失，32、64 帧目标的相对权重为 2 和 1.5。可用 `--block-size` 设置块长度。

长程实验按验证场景的整段 ATE、前 120 帧对齐后的后缀平移 RMSE 与长间隔旋转 RPE 选模。各项误差除以原版 12 帧在相同场景的对应指标，再取等权平均。

相同数据下，长程训练按综合误差选中第 1 轮。两条 1000 帧测试中，相对首轮微调，`scene0140_00` 的 ATE 从 17.46 降到 16.76 厘米，500 帧旋转 RPE 从 12.15° 降到 10.72°。`scene0070_00` 的 ATE 从 22.27 升到 26.24 厘米，500 帧旋转 RPE 从 12.03° 降到 11.61°。

### 使用历史共视点的视觉 TTT

`--visual-ttt` 从 `--visual-gate-weights` 指定的六参数视觉门启动。冻结其余模型权重，每隔 8 帧尝试从当前视觉上下文中的窗外帧提取通过 PnP 核验的点对应。历史点由模型先前的点图和位姿获得，当前射线取自初次前向的点图。按图像网格划分梯度和核验两组，用角度残差对六参数做一次限幅更新。用相同相机状态重放当前输入，核验组误差下降后，候选权重从下一帧生效。

```bash
python -m stream_reconstruction "$RGB" --checkpoint "$CHECKPOINT" \
  --output runs/visual-online --pose-mode baseline --visual-context structured24 \
  --visual-gate-weights "$GATE_WEIGHTS" --visual-ttt --profile
```

对照使用相同权重和输入，省略 `--visual-ttt` 即可。`trajectory.jsonl` 的 `visual_ttt` 记录触发、参考和接受状态。每次候选重放计入 `--profile` 延迟。

### 前缀快权重

`--prefix-fast-weights` 收集序列前 12 帧最后一层跨帧注意力的 K/V。先用前 10 帧按注意力头拟合带 L2 正则的 K→V 线性映射，再用第 11、12 帧比较拟合误差与均值预测。通过后用 12 帧重拟合一次，后续帧的 Q 从固定大小的快权重读出，并以 5% 比例融合到已有的视觉门输出。K 和 Q 统一时间 RoPE 相位。 [tttLRM 的快权重读写机制](https://arxiv.org/html/2602.20160v2#S3)

```bash
python -m stream_reconstruction "$RGB" --checkpoint "$CHECKPOINT" \
  --output runs/visual-prefix --pose-mode baseline --visual-context structured24 \
  --visual-gate-weights "$GATE_WEIGHTS" --prefix-fast-weights --profile
```

`trajectory.jsonl` 的 `prefix_calibration` 记录留出帧拟合结果，`prefix_fast_weights_applied` 标记使用快权重的帧。拟合和读取都用测试序列的视觉张量，无需外部相机位姿。

## 选帧规则

1. 单次提取空间 patch token，做平均池化并进行 L2 归一化。每帧最多采集 192 个 ORB 特征，读取对应像素的预测局部三维点和置信度，将局部射线旋转到共同的世界坐标系。
2. 候选帧在 `--lookahead` 步后到期。检查其有效点的网格覆盖，并在紧邻它的下一帧与当前到达帧中寻找可靠的共视支持。视野切换时，得到后续支持的新区域可以建锚。
3. 在记忆中按池化向量的余弦相似度检索最多两个参考锚，再做稀疏描述子匹配。已有锚覆盖该区域且射线夹角较小时跳过。共视充分、角度有变化或出现稳定新区域时保存候选。满额时释放最早的锚。

默认观察 4 个后续输入，长期容量为 8 张，帧间最小间隔为 1 步。参考锚的共视覆盖阈值为 0.45，中位射线夹角阈值为 2 度。点置信度下限沿用 ABot 推理接口的默认值 0，可用 `SelectionConfig.min_confidence` 调整。时间、容量和几何阈值同样可配置。这些是实现初值，长序列的精度与速度需要单独评估。

## 可选的旧深度修订

使用 `--revise-depth` 时保持 `--lookahead 4`。旧帧被选中后，利用随后四张已到达图像做有置信度加权的几何融合。输出 `depth_000000.npy` 等文件，编号对应候选帧，数组是原输入每四个像素采样的局部相机深度。该选项供比较既有的固定融合。在线更新网络参数的 TTT 属于后续研究。

已有的光流选锚对照也可以独立回放。安装对应版本的 VGGT-SLAM 后，将有序图像路径和一个新建的 `FrameTracker` 传给下列函数。它在相同的 504×280 输入上调用作者原版的 50 像素判据。[对照使用的作者版本](https://github.com/MIT-SPARK/VGGT-SLAM/blob/d4d4ae2ecfaf1101b46e43b770c956407ed79146/vggt_slam/frame_overlap.py)

```python
from stream_reconstruction.abot import motion_keyframes
from vggt_slam.frame_overlap import FrameTracker

indices = list(motion_keyframes(paths, FrameTracker()))
```

## 窗外历史帧的首发位姿实验

每次运行生成 `trajectory.jsonl`，每个实际输入对应一行。`raw` 和 `pose` 分别记录主干和发布的相机到世界 4×4 矩阵。Python 接口从 `StreamStep.published_pose` 读取发布位姿。用局部点图生成修正点云时，对每个局部点应用该位姿。模型预测字典仍使用原网络的坐标。

| `--pose-mode` | 当前帧可读取的几何参考 |
| --- | --- |
| `baseline`（默认） | 直接发布所选视觉窗口的主干预测 |
| `anchors` | 额外从已提交锚中按余弦相似度检索最多 6 张窗外帧 |
| `fixed` | 在历史锚之外保留最初 3 张真实帧，离开近期窗口后可用，按帧号去重 |

两种修正模式逐候选用 ORB 匹配和归一化像面 PnP-RANSAC 核对内点覆盖、正深度、尺度与位姿更新幅度。默认 `--pose-solver single` 采用内点最多的参考，将旋转和平移各融合半步。`--pose-solver joint` 至少需要两个通过检查的参考，使用它们的内点做一次六自由度稳健高斯–牛顿求解。重复出现的当前特征共享观测权重，平移步长按中位深度归一化。联合更新须通过可观测性、幅度和带先验目标下降的检查。没有可用更新时，沿上一发布位姿累积原始相邻变换。

`trajectory.jsonl` 中，`candidates` 记录检索结果，`verified` 记录通过 PnP 检查的参考，`references` 记录实际采用的参考。`inliers` 记录参与发布计算的不同当前特征数。Python 接口使用 `pose_solver="joint"` 启用联合求解。

在同一新输出根目录下复现实验，可分别运行三种模式。以下命令从仓库根目录执行，`RGB` 和 `CHECKPOINT` 分别指向图像目录和本地权重。

```bash
for variant in baseline:single anchors:single fixed:single anchors:joint fixed:joint; do
  mode=${variant%:*}
  solver=${variant#*:}
  python -m stream_reconstruction "$RGB" --visual-context local12 --checkpoint "$CHECKPOINT" \
    --pose-mode "$mode" --pose-solver "$solver" --output "runs/$mode-$solver" || break
done
```

2026 年 11 月 22 日在 RTX 3090、BF16、SDPA 上，使用 ABot-Recon 版本 `6e3b2e1` 对 ScanNet v2 的 `scene0000_00` 起始连续 120 帧（0–119）各运行一次。三组原始预测位姿逐元素一致。用对应帧的 ScanNet 相机中心作参考，按整段 Sim(3) 对齐后计算平移 RMSE。

| 模式 | 采用几何修正的帧数 | ATE RMSE |
| --- | ---: | ---: |
| baseline | 0 | 2.804 厘米 |
| anchors | 91 | 3.534 厘米 |
| fixed | 92 | 3.034 厘米 |

该片段上，两种修正都增大了误差。三万帧漂移、跨场景泛化和点云精度尚待独立评测。

### 两场景联合求解对照

相同权重和推理设置下，使用 ScanNet 的 `scene0000_00` 前 600 帧与 `scene0062_00` 前 300 帧。每条序列运行五个配置，五组主干位姿逐元素一致，锚帧决策也一致。每个单元格依次给出整段 Sim(3) 对齐后的 ATE RMSE，以及只用前 120 帧拟合 Sim(3) 后计算的后缀平移 RMSE，单位为厘米。参考位姿取自同一传感器文件，帧间误差按帧数计算。

| 配置 | scene0000_00（600 帧） | scene0062_00（300 帧） |
| --- | ---: | ---: |
| 基线 | 20.02 / 38.72 | 4.49 / 16.68 |
| 历史锚，单参考 | 8.93 / **12.74** | 4.55 / 15.72 |
| 历史锚与固定帧，单参考 | **8.60** / 18.22 | 4.05 / 13.29 |
| 历史锚，联合 | 14.44 / 28.23 | **3.70** / **8.83** |
| 历史锚与固定帧，联合 | 14.45 / 28.38 | 3.86 / 9.85 |

仅历史锚的联合求解在两个片段中都降低了相对基线的后缀平移误差，最佳求解器随场景而变。120 帧间隔的相对旋转 RMSE 分别从基线的 2.86°、2.83° 降至 2.36°、1.71°。这两条片段共 900 帧，长序列和跨场景收益仍需更广泛验证。

### 三槽平面方向摘要

使用 `--pose-mode fixed --pose-solver joint` 时，可再指定 `--summary off|recent|full`。Python 接口使用同名 `summary` 参数。

| 模式 | 三个方向槽的观测范围 |
| --- | --- |
| `off`（默认） | 几何联合求解使用锚帧与固定帧 |
| `recent` | 最近三个已结束的 24 帧子图，保留与全历史相同的方向查询和槽位容量 |
| `full` | 所有已结束且通过对齐核验的子图 |

局部点图按 8 像素间隔采样，在图像网格内用协方差拟合平面。每 4 帧最多为当前子图采集一次，首个子图提供坐标起点，之后要求存在通过 PnP 检查的跨子图对应。子图结束时按法向外积提交三个固定查询的加权统计。每个方向须在两个不同帧中出现，读取时须有两个以上子图支持且角度离散度通过检查。摘要在下一帧开放。当点约束不足时，方向可以独立修正旋转。

日志的 `directions` 记录用于发布位姿的槽号、独立子图数和方向离散度，`summary_commit` 记录本帧结束时新提交的子图，`summary_submaps` 记录提交和过期处理后的子图数。全历史只在求解时读取三个汇总条目，同时保存各子图的矩阵和权重贡献供重新对齐时替换。当前流保存已发布首发轨迹，全局子图重定位与 TTT 属于后续阶段。

使用与联合求解对照相同的两段 ScanNet 图像和同一套权重。每段运行三个摘要模式，统一保留固定帧和历史锚，冻结主干输出与选锚决定一致。

| 场景 | 摘要模式 | 整段 ATE RMSE（厘米） | 前 120 帧对齐后的后缀 RMSE（厘米） | 120 帧间隔旋转 RMSE（度） | 采用方向的帧数 |
| --- | --- | ---: | ---: | ---: | ---: |
| scene0000_00（600 帧） | off | 14.45 | 28.38 | 2.35 | 0 |
| scene0000_00（600 帧） | recent | 13.41 | 30.63 | 6.22 | 99 |
| scene0000_00（600 帧） | full | 19.19 | 39.35 | 7.09 | 264 |
| scene0062_00（300 帧） | off | 3.86 | 9.85 | 1.76 | 0 |
| scene0062_00（300 帧） | recent | 3.78 | 10.51 | 1.84 | 235 |
| scene0062_00（300 帧） | full | 3.86 | 10.99 | 2.12 | 241 |

两个片段中，关闭方向项时的后缀平移和 120 帧间隔旋转误差最低。全历史摘要比近期摘要触发更多帧，且引入了更大的后缀平移和旋转误差。在全历史模式只用方向更新的 189 帧和 141 帧上，相对于同一帧位姿先验的原始相邻旋转，地面参考下的相邻旋转 RMSE 分别从 0.233° 上升至 0.608°、从 0.206° 上升至 0.273°。这些数据覆盖两个室内短片段，默认关闭方向摘要。

### 方向权重与写入共识

`--direction-gain` 缩放联合求解中方向残差的能量，接受 0 至 1，默认 1。比较关闭摘要或启用弱方向约束时，可以使用下列命令中的相应参数。

```bash
python -m stream_reconstruction "$RGB" --visual-context local12 --checkpoint "$CHECKPOINT" \
  --pose-mode fixed --pose-solver joint --summary full --direction-gain 0.1 \
  --output runs/full-light
```

加上 `--summary-alignment consensus` 后，首块之后的写入样本需要两个来自不同历史子图的有效 PnP 参考，它们给出的旋转相差最多 2°。在满足条件的参考对中优先取较弱成员内点数最多的一对，然后按内点数平均旋转向量。缺少共识时跳过该样本。默认 `--summary-alignment single` 仍按内点最多的参考写入。两项参数也可通过 Python 接口的 `direction_gain` 和 `summary_alignment` 设置。

固定先验回放在两个开发片段上比较 0 至 1 的多个权重。选用 0.1 做闭环对照，写入共识阈值在独立场景评估前固定。下表每格依次为用前 120 帧对齐后的后缀平移 RMSE（厘米）和 120 帧间隔的相对旋转 RMSE（度）。

| 场景 | 关闭摘要 | 原全历史摘要 | 权重 0.1 | 权重 0.1 ＋写入共识 |
| --- | ---: | ---: | ---: | ---: |
| scene0000_00，600 帧 | 28.38／2.35 | 39.35／7.09 | 27.44／2.82 | 27.31／2.28 |
| scene0062_00，300 帧 | 9.85／1.76 | 10.99／2.12 | 10.48／1.54 | 10.53／1.51 |
| scene0070_00，300 帧，独立场景 | 30.00／4.27 | 28.61／4.47 | 31.35／4.02 | 30.90／4.12 |

降权加共识比原全历史摘要在三个片段上都降低了旋转误差。独立场景的平移误差比关闭摘要高 0.90 厘米，在使用共识写入时仍有 120 帧用到方向约束。当前默认关闭摘要。

## 在线融合门与子图输出

在固定帧、联合求解和全历史摘要模式下，`--ttt fixed` 和 `--ttt online` 使用同一组几何阈值、两门初值以及对应分组规则。带偏置的两个线性门各读取两项质量指标，共六个参数。点对应门从 0.95 启动，方向门从 `--direction-gain` 启动，后者须在 0 和 1 之间。

当前特征继承已匹配参考特征的分组，冲突的轨迹退出全部分组。求解只读取历史锚和固定帧的求解组。两个近期帧可补充在线训练组与核验组的对应。每隔 8 帧，在具有足够内点、覆盖和可观测性的情况下，对一次高斯–牛顿更新求导。梯度只传给六个门参数。根据训练角残差提出权重候选，再用独立对应核验，接受后从下一帧开始使用。日志 `ttt` 记录两门权重、触发、接受和下一帧生效标记。

```bash
python -m stream_reconstruction "$RGB" --visual-context local12 --checkpoint "$CHECKPOINT" --output runs/online \
  --pose-mode fixed --pose-solver joint --summary full --direction-gain 0.1 \
  --summary-alignment consensus --ttt online --profile --submaps
```

`--profile` 保存首发和处理到写入完成的 P95 延迟、CUDA 分配与预留峰值及索引数值数组占用。延迟从开始处理一帧计时，到交付该帧的位姿或本轮输出写入结束。GPU 缓存、Python 对象和点云文件在各自对应的统计口径下单独计算。

`--submaps` 每 24 帧开一个子图，每 10 帧保存一次按像素间隔 8 采样的局部点。点文件按最初发布姿态进入子图坐标，`submaps.json` 提供拼接的世界变换。若同时指定 `--loop-salad-checkpoint` 与 `--loop-dino-checkpoint`，并安装上游 ABot-Recon 的 `loop` 依赖，则流结束后调用官方回环。原始主干的回环轨迹保存为 `abot_loop_poses.npy`，回环修正应用到子图锚帧，新的世界变换保存到 `submaps_loop.json`。离线回环结果和因果首发轨迹各自可读取。

### ScanNet 在线训练对照

| 片段 | 固定门整段 ATE | 在线整段 ATE | 在线接受／触发 |
| --- | ---: | ---: | ---: |
| 600 帧 | 13.603 厘米 | 13.603 厘米 | 1／16 |
| 1000 帧 | 20.422 厘米 | 20.424 厘米 | 11／21 |
| 5000 帧 | 32.276 厘米 | 32.277 厘米 | 41／101 |

目前未观察到在线门相对固定门的精度收益。5000 帧时，固定门与在线门的端到端 P95 分别为 155.0 和 157.2 毫秒，CUDA 峰值实际分配均约 7.38 GiB。该结果来自室内连续序列。

## 监督式离线训练融合门

冻结 ABot 权重，用有参考相机位姿的序列训练点对应门和历史方向门。两门各学习三个系数，根据当前帧的质量指标计算残差权重。训练对带权重的一次高斯–牛顿更新求导，由真实相邻位姿监督，按验证场景的损失选择权重。推理逐帧读取已训练的系数并按输入计算权重，无需参考位姿或梯度更新。

先为每个训练和验证场景采集一次。各场景使用独立的输出目录。

```bash
python -m stream_reconstruction "$RGB" --visual-context local12 --checkpoint "$CHECKPOINT" --output runs/scene-data \
  --pose-mode fixed --pose-solver joint --summary full --direction-gain 0.1 \
  --summary-alignment consensus --ttt fixed --capture-gates
```

采集生成 `gate_evidence.npz` 及相邻的 `trajectory.jsonl`。参考文件为含 `poses` 的 NPZ，保存与 RGB 顺序严格对应的相机到世界 4×4 矩阵，缺少参考的帧用 NaN 表示。尺度从序列前 120 帧的相机中心估计，起始位置需包含足够的非共线运动。以下清单字段的相对路径均基于清单所在目录。

```json
{
  "direction_gain": 0.1,
  "train": [
    {"scene": "train-scene", "evidence": "runs/train/gate_evidence.npz", "reference": "data/train/reference.npz"}
  ],
  "validation": [
    {"scene": "validation-scene", "evidence": "runs/validation/gate_evidence.npz", "reference": "data/validation/reference.npz"}
  ]
}
```

```bash
OMP_NUM_THREADS=1 python -m stream_reconstruction.train_gates manifest.json --output runs/model
python -m stream_reconstruction "$TEST_RGB" --visual-context local12 --checkpoint "$CHECKPOINT" --output runs/trained \
  --pose-mode fixed --pose-solver joint --summary full --direction-gain 0.1 \
  --summary-alignment consensus --ttt fixed --gate-weights runs/model/gates.json
```

`history.json` 保存损失、最佳轮次以及各场景的证据与参考文件哈希。若初始权重取得最低验证损失，最佳轮次为 0，保存初始权重。`trajectory.jsonl` 的 `ttt` 字段记录每帧实际使用的两门权重。

在 ScanNet v2 上，使用三个场景训练和一个独立场景选模，选中第 120 轮，单步验证损失下降 7.2%。未参与训练和选模的 `scene0140_00` 前 1000 帧，冻结主干、固定门和训练门的整段 Sim(3) ATE 分别为 16.67、15.42 和 15.44 厘米。此前研究过的另一场景 `scene0070_00` 前 1000 帧，固定门和训练门的 ATE 分别为 20.42 和 24.82 厘米。基于这些结果，训练权重作为实验选项手动加载。
