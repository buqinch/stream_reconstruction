"""在 ABot 单次流式前向中取出可用于选帧的稀疏观测。"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .directions import plane_axes
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


def reconstruct(network, paths, config=SelectionConfig(), *, revise_depth=False,
                pose_mode="baseline", pose_solver="single", summary="off",
                direction_gain=1., summary_alignment="single", ttt="off", profile=False,
                gate_parameters=None, training_evidence=None, visual_context="structured24",
                visual_gate=None, gradient_interval=0, visual_ttt=False,
                prefix_fast_weights=False, history_fast_weights=False):
    """逐帧返回原模型预测及到期决定，首版预测由调用方按需消费。"""
    import torch
    from abot_recon.modeling.streaming.core.cache_utils import detach_carry

    from .visual_context import RECENT, stream_network

    if revise_depth and config.lookahead != DELAY:
        raise ValueError("既有深度融合需要四个后续输入")
    if pose_mode == "baseline" and pose_solver != "single":
        raise ValueError("联合求解需要历史锚或固定帧模式")
    if pose_mode == "baseline" and (summary != "off" or ttt != "off" or
                                    gate_parameters is not None or training_evidence is not None):
        raise ValueError("方向摘要需要固定帧和联合求解模式")
    device = next(network.parameters()).device
    if gradient_interval < 0 or (gradient_interval and visual_gate is None):
        raise ValueError("逐帧反传需要视觉门及正的采样间隔")
    adaptation = None
    if sum((prefix_fast_weights, history_fast_weights, visual_ttt)) > 1 or (
            (prefix_fast_weights or history_fast_weights) and gradient_interval):
        raise ValueError("快权重、视觉 TTT 与训练需独立运行")
    if visual_ttt:
        from copy import copy, deepcopy

        from .visual_ttt import VisualTTT

        if visual_gate is None or gradient_interval or any(p.requires_grad for p in network.parameters()):
            raise ValueError("视觉 TTT 需要冻结主干、初始视觉门和独立的在线反传时序")
        visual_gate = deepcopy(visual_gate).requires_grad_(True)
        adaptation = VisualTTT(visual_gate)
        gradient_interval = adaptation.interval
    network, visual = stream_network(network, config, visual_context, visual_gate,
                                     prefix_fast_weights=prefix_fast_weights,
                                     history_fast_weights=history_fast_weights)
    selector = AnchorSelector(config)
    memory = None if pose_mode == "baseline" else PoseMemory(
        pose_mode, config.capacity, RECENT, solver=pose_solver, summary=summary,
        direction_gain=direction_gain, alignment=summary_alignment, ttt=ttt,
        gate_parameters=gate_parameters, training_evidence=training_evidence)
    detector = cv2.ORB_create(nfeatures=192)
    geometry = deque(maxlen=DELAY + 1)
    carry = reference = camera_state = None

    for index, path in enumerate(paths):
        tensor, rgb = _input_image(path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gradient = index >= RECENT and index % gradient_interval == 0 if gradient_interval else False
        if adaptation is not None and gradient:
            initial_context = copy(visual)
            initial_state, initial_reference = camera_state, reference
        grad_context = torch.set_grad_enabled(gradient) if gradient_interval else torch.inference_mode()
        with grad_context, torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            frame = tensor[None, None].to(device=device, dtype=dtype)
            if visual is None:
                prediction = network(
                    frame, causal_global_attn=True, long_sequence_parallel=True,
                    streaming_inference=True, past_key_values=carry,
                    ref_hidden=reference, camera_state=camera_state)
                carry = network._postprocess_stream_carry(
                    detach_carry(prediction.get("past_key_values")), spatial_hw=gray.shape)
            else:
                prediction = network._forward_frame_paged(
                    frame, frame_idx=index, ref_hidden=reference, camera_state=camera_state)
            reference = prediction.get("ref_hidden", reference)
            camera_state = prediction["camera_state"]
        descriptor = network.encoder.descriptor
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
            values = points[rows, cols].detach().float().cpu().numpy()
            weights = confidence[rows, cols].detach().float().sigmoid().cpu().numpy()
            return values, weights

        observation = observe(
            index, descriptor.cpu().numpy(), gray, pose, detector, sample_geometry,
            config.min_confidence)
        decision = selector.push(observation)
        published, evidence = pose, {}
        if memory is not None:
            axes = ()
            if summary != "off":
                grid = points[::8, ::8].float().cpu().numpy()
                scores = confidence[::8, ::8].float().sigmoid().cpu().numpy()
                axes = plane_axes(grid, scores, max(0.25, config.min_confidence))
            published, evidence = memory.advance(observation, pose, decision, axes)
            if profile:
                anchor_bytes, summary_bytes = memory.index_payload_bytes()
                evidence.update(anchor_index_bytes=anchor_bytes, summary_index_bytes=summary_bytes)
        if visual is not None:
            if adaptation is not None:
                update = dict(accepted=False)
                if gradient:
                    def replay():
                        network._paged_manager = network.encoder.context = initial_context
                        try:
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
                                candidate = network._forward_frame_paged(
                                    frame, frame_idx=index, ref_hidden=initial_reference,
                                    camera_state=initial_state)
                            return candidate["camera_poses"][0, 0].float()
                        finally:
                            network._paged_manager = network.encoder.context = visual

                    update = adaptation.update(
                        observation, pose, prediction["camera_poses"][0, 0].float(),
                        visual.trace["anchors"] + visual.trace["fixed"], replay)
                evidence["visual_ttt"] = update
            with torch.no_grad():
                visual.finish(decision)
            if adaptation is not None:
                adaptation.remember(observation, pose, visual)
            evidence["visual_context"] = visual.trace
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
