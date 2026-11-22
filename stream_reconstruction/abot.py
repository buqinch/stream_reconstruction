"""在 ABot 单次流式前向中取出可用于选帧的稀疏观测。"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .trajectory import PoseMemory
from .fusion import DELAY, Frame, revise
from .selection import AnchorSelector, Decision, SelectionConfig, observe


@dataclass
class StreamStep:
    index: int
    prediction: dict | None = None
    decision: Decision | None = None
    revised_depth: np.ndarray | None = None
    published_pose: np.ndarray | None = None
    pose_evidence: dict | None = None


def _input_image(path):
    from abot_recon.preprocessing import preprocess_image

    with Image.open(path) as image:
        tensor, _ = preprocess_image(image.convert("RGB"), height=280, width=504)
    rgb = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return tensor, rgb


def motion_keyframes(paths, tracker, threshold=50):
    """用单个序列专用的 VGGT-SLAM FrameTracker 复现稀疏光流选帧。"""
    for index, path in enumerate(paths):
        _, rgb = _input_image(path)
        if tracker.compute_disparity(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), threshold):
            yield index


def reconstruct(network, paths, config=SelectionConfig(), *, revise_depth=False, pose_mode="baseline"):
    """逐帧返回原模型预测及到期决定，首版预测由调用方按需消费。"""
    import torch
    from abot_recon.modeling.streaming.core.cache_utils import detach_carry

    if revise_depth and config.lookahead != DELAY:
        raise ValueError("既有深度融合需要四个后续输入")
    device = next(network.parameters()).device
    selector = AnchorSelector(config)
    memory = None if pose_mode == "baseline" else PoseMemory(
        pose_mode, config.capacity, network.local_window_frames)
    detector = cv2.ORB_create(nfeatures=192)
    geometry = deque(maxlen=DELAY + 1)
    carry = reference = camera_state = descriptor = None

    def pool(_module, _args, output):
        nonlocal descriptor
        tokens = output["x_norm_patchtokens"] if isinstance(output, dict) else output
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError("选帧器需要单张图像的空间 patch token")
        descriptor = tokens.detach().mean(dim=1, dtype=torch.float32)[0]

    hook = network.encoder.register_forward_hook(pool)
    try:
        for index, path in enumerate(paths):
            tensor, rgb = _input_image(path)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            descriptor = None
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                prediction = network(
                    tensor[None, None].to(device=device, dtype=dtype), causal_global_attn=True,
                    long_sequence_parallel=True, streaming_inference=True,
                    past_key_values=carry, ref_hidden=reference, camera_state=camera_state,
                )
                carry = network._postprocess_stream_carry(
                    detach_carry(prediction.get("past_key_values")), spatial_hw=gray.shape)
                reference = prediction.get("ref_hidden", reference)
                camera_state = prediction["camera_state"]
            if descriptor is None or prediction.get("conf") is None:
                raise ValueError("需要提供编码器 patch token 和置信度的 ABot 模型")
            points = prediction["local_points"][0, 0]
            confidence = prediction["conf"][0, 0, :, :, 0]
            pose = prediction["camera_poses"][0, 0].detach().float().cpu().numpy()
            if points.shape != (*gray.shape, 3) or tuple(confidence.shape) != gray.shape:
                raise ValueError("模型点图和置信度须与输入像素对齐")

            def sample_geometry(y, x):
                rows = torch.as_tensor(y, device=device)
                cols = torch.as_tensor(x, device=device)
                values = points[rows, cols].float().cpu().numpy()
                weights = confidence[rows, cols].float().sigmoid().cpu().numpy()
                return values, weights

            observation = observe(
                index, descriptor.cpu().numpy(), gray, pose, detector, sample_geometry,
                config.min_confidence)
            decision = selector.push(observation)
            published, evidence = pose, {}
            if memory is not None:
                published, evidence = memory.advance(observation, pose, decision)
            corrected = None
            if revise_depth:
                geometry.append(Frame(
                    index, gray, points[::4, ::4].float().cpu().numpy(),
                    confidence[::4, ::4].float().sigmoid().cpu().numpy(), pose))
                if decision is not None and decision.kept:
                    corrected, _ = revise(list(geometry))
            yield StreamStep(index, prediction, decision, corrected, published, evidence)
        for decision in selector.finish():
            yield StreamStep(decision.decided_at, decision=decision)
    finally:
        hook.remove()
