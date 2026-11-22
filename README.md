# 基于 ABot-Recon 的流式锚帧选择

对单目 RGB 视频逐帧输出 ABot 的三维预测。选择器观察若干张已经到达的后续帧，再决定是否把候选帧存为长期锚。每次选择均记录候选帧、决定发生时的输入帧、参考锚、覆盖率、视差和选择原因。候选与锚帧的序号从 0 开始。

## 运行

使用已安装官方 ABot-Recon、PyTorch、Pillow、NumPy 和 OpenCV 的 Python 环境。命令行使用 CUDA 和 SDPA。输入可以是按文件名排序的 RGB 目录，也可以是 TUM 的 `rgb.txt`，后者每行给出时间戳和相对路径。可指定只含所需采样帧的清单。输出目录须为新目录。[ABot-Recon 安装与权重](https://github.com/amap-cvlab/ABot-Recon)

```bash
python -m stream_reconstruction rgb.txt --checkpoint /path/to/abot_recon.safetensors --output run-01
python -m unittest discover -s tests -v
```

`decisions.jsonl` 中每个输入帧均有一条决定。项目也提供 `stream_reconstruction.abot.reconstruct(network, paths, config)`，每个输入步返回原模型的预测字典。流结束后还会发送尾帧的决定，此时 `prediction` 为空。每条流使用独立的模型编码器钩子，提前停止迭代时应关闭生成器。

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
| `baseline`（默认） | 原版冻结主干的近期窗口 |
| `anchors` | 额外从已提交锚中按余弦相似度检索最多 6 张窗外帧 |
| `fixed` | 在历史锚之外保留最初 3 张真实帧，离开近期窗口后可用，按帧号去重 |

两种修正模式逐候选用 ORB 匹配和归一化像面 PnP-RANSAC 核对内点覆盖、正深度、尺度与位姿更新幅度。默认 `--pose-solver single` 采用内点最多的参考，将旋转和平移各融合半步。`--pose-solver joint` 至少需要两个通过检查的参考，使用它们的内点做一次六自由度稳健高斯–牛顿求解。重复出现的当前特征共享观测权重，平移步长按中位深度归一化。联合更新须通过可观测性、幅度和带先验目标下降的检查。没有可用更新时，沿上一发布位姿累积原始相邻变换。

`trajectory.jsonl` 中，`candidates` 记录检索结果，`verified` 记录通过 PnP 检查的参考，`references` 记录实际采用的参考。`inliers` 记录参与发布计算的不同当前特征数。Python 接口使用 `pose_solver="joint"` 启用联合求解。三个全历史摘要和 TTT 仍待研究。

在同一新输出根目录下复现实验，可分别运行三种模式。以下命令从仓库根目录执行，`RGB` 和 `CHECKPOINT` 分别指向图像目录和本地权重。

```bash
for variant in baseline:single anchors:single fixed:single anchors:joint fixed:joint; do
  mode=${variant%:*}
  solver=${variant#*:}
  python -m stream_reconstruction "$RGB" --checkpoint "$CHECKPOINT" \
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
