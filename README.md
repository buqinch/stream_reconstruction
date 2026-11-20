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
