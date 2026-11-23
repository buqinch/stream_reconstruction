"""为 ABot 的逐层跨帧注意力保存并选择视觉 K/V。"""

from collections import deque
from copy import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import scaled_dot_product_attention


RECENT = 12
ANCHORS = 6
FIXED = 3
SUMMARIES = 3


class FrameEncoder(torch.nn.Module):
    """在当前图像编码完成后、跨帧解码之前选择视觉历史。"""

    def __init__(self, encoder, context):
        super().__init__()
        self.encoder = encoder
        self.context = context
        self.descriptor = None

    def forward(self, *args, **kwargs):
        output = self.encoder(*args, **kwargs)
        tokens = output["x_norm_patchtokens"] if isinstance(output, dict) else output
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError("选帧器需要单张图像的空间 patch token")
        self.descriptor = tokens.detach().mean(dim=1, dtype=torch.float32)[0]
        if self.context is not None:
            self.descriptor = self.descriptor.cpu()
            self.context.begin(self.descriptor.numpy())
        return output


def stream_network(network, config, mode, visual_gate=None, *,
                   prefix_fast_weights=False, history_fast_weights=False):
    if mode not in ("structured24", "local12"):
        raise ValueError("视觉窗口须选择 structured24 或 local12")
    if network.local_window_frames != RECENT:
        raise ValueError("请从发布模型的 12 帧配置启动视觉窗口实验")
    context = None
    if visual_gate is not None and mode != "structured24":
        raise ValueError("视觉门需要 structured24 模式")
    if (prefix_fast_weights or history_fast_weights) and visual_gate is None:
        raise ValueError("快权重需要视觉门")
    if mode == "structured24":
        if network.global_pos_encoding != "rope3d" or network.use_global_points:
            raise ValueError("结构化窗口需要发布模型的三维 RoPE 和局部点图头")
        context = VisualContext(network.num_global_decoder_blocks, network.rope3d_embed,
                                config, visual_gate=visual_gate,
                                prefix_fast_weights=prefix_fast_weights,
                                history_fast_weights=history_fast_weights)
    # 每条流有自己的编码器入口和缓存管理器。注册表浅拷贝，参数及预测头共享。
    runtime = copy(network)
    runtime._modules = network._modules.copy()
    runtime.encoder = FrameEncoder(network.encoder, context)
    if context is not None:
        runtime.local_window_frames = RECENT + ANCHORS + FIXED + SUMMARIES
        runtime._paged_manager = context
    return runtime, context


@dataclass
class VisualFrame:
    index: int
    descriptor: np.ndarray
    kv: torch.Tensor | None = None


class VisualSummary:
    """同一视觉簇的 K/V 均值，K 的时间相位统一到首个成员。"""

    def __init__(self, frame):
        self.first = self.last = frame.index
        self.count = 1
        self.center = frame.descriptor.copy()
        self.mean = frame.kv.to(torch.float32, copy=True)
        self.kv = frame.kv

    def add(self, frame, rope):
        from abot_recon.modeling.streaming.rope3d import apply_rotary_emb

        sample = frame.kv.to(torch.float32, copy=True)
        time_dim = rope.fhw_dim[0]
        # 两个绝对时刻的相位差只作用于时间通道，保留每个 patch 的空间位置。
        phase = (rope.freqs[self.first, :time_dim // 2]
                 * rope.freqs[frame.index, :time_dim // 2].conj())
        keys = sample[:, 0, ..., :time_dim]
        keys.copy_(apply_rotary_emb(keys, phase))
        self.count += 1
        self.mean.lerp_(sample, 1.0 / self.count)
        self.kv = self.mean.to(frame.kv.dtype)
        self.center += (frame.descriptor - self.center) / self.count
        self.last = frame.index


class VisualContext:
    """单条流的缓存管理器，接口对应上游 paged_stream_forward 的两个回调。"""

    def __init__(self, layers, rope, config, *, visual_gate=None,
                 prefix_fast_weights=False, history_fast_weights=False):
        self.layers = layers
        self.visual_gate = visual_gate
        self.prefix = None
        self.history_fast = None
        if prefix_fast_weights:
            from .prefix_fast_weights import PrefixFastWeights

            self.prefix = PrefixFastWeights(rope)
        if history_fast_weights:
            from .prefix_fast_weights import HistoricalFastWeights

            self.history_fast = HistoricalFastWeights(rope)
        self.rope = rope
        self.lookahead = config.lookahead
        self.recent = deque()
        self.fixed = []
        self.anchors = deque(maxlen=config.capacity)
        self.pending = {}
        self.admitted = set()
        self.summaries = []
        self.next_index = 0
        self.current = None
        self.selected = []
        self.trace = {}

    def begin(self, descriptor):
        if self.current is not None:
            raise RuntimeError("上一帧尚未完成")
        index = self.next_index
        if index >= self.rope.max_seq_len:
            raise ValueError("帧号超出 RoPE 时间表，请在加载模型时增大 max_frames")
        norm = np.linalg.norm(descriptor)
        if descriptor.ndim != 1 or not np.isfinite(norm) or norm == 0:
            raise ValueError("视觉检索需要非零有限帧描述子")
        self.current = VisualFrame(index, descriptor / norm)
        self.written_layers = 0
        anchors = sorted(self.anchors, key=lambda f: float(f.descriptor @ self.current.descriptor),
                         reverse=True)[:ANCHORS]
        explicit = [*self.fixed, *anchors, *self.recent]
        real_ids = {f.index for f in explicit}
        # 单例摘要可能恰好就是某个固定帧或召回锚，此时只读一次。
        active = [(i, group) for i, group in enumerate(self.summaries)
                  if group.count > 1 or group.first not in real_ids]
        self.selected = [f.kv for f in explicit] + [g.kv for _, g in active]
        if self.history_fast is not None:
            self.history_fast_entries = [((0, f.index, 0), f.kv, f.index)
                                         for f in (*self.fixed, *anchors)]
            # 摘要的键已旋转到其首个成员的时间相位。
            self.history_fast_entries.extend(((1, slot, group.count), group.kv, group.first)
                                             for slot, group in active)
        self.trace = dict(recent=[f.index for f in self.recent] + [index],
                          anchors=[f.index for f in anchors], fixed=[f.index for f in self.fixed],
                          summaries=[dict(slot=i, count=g.count, first=g.first, last=g.last)
                                     for i, g in active],
                          positions=len(self.selected) + 1)
        if self.visual_gate is not None:
            self.history_groups = (
                [f.kv for f in self.fixed], [f.kv for f in anchors], [g.kv for _, g in active])
            descriptors = ([f.descriptor for f in self.fixed], [f.descriptor for f in anchors],
                           [g.center for _, g in active])
            self.similarity = [
                max((float(item @ self.current.descriptor) / max(np.linalg.norm(item), 1e-8)
                     for item in group), default=0.)
                for group in descriptors
            ]

    def append_frame(self, layer_idx, keys, values):
        if self.current is None or layer_idx != self.written_layers or layer_idx >= self.layers:
            raise RuntimeError("视觉 K/V 须按当前帧的解码层顺序写入")
        if layer_idx == 0:
            self.current.kv = keys.new_empty((self.layers, 2, *keys.shape))
        layer = self.current.kv[layer_idx]
        layer[0].copy_(keys.detach())
        layer[1].copy_(values.detach())
        self.written_layers += 1

    @staticmethod
    def attend(query, cache):
        keys, values = cache.unbind(0)
        attended = scaled_dot_product_attention(
            query.transpose(0, 1)[None], keys.transpose(0, 1)[None], values.transpose(0, 1)[None])
        return attended[0].transpose(0, 1)

    def compute_attention(self, layer_idx, query):
        # 上游已完成 Q/K 归一化和真实时间的三维 RoPE，每槽有一整帧 token。
        current = self.current.kv[layer_idx]
        if self.visual_gate is not None:
            recent = torch.cat([f.kv[layer_idx] for f in self.recent] + [current], dim=1)
            local = self.attend(query, recent)
            if layer_idx != self.layers - 1:
                return local
            weights = self.visual_gate(torch.tensor(self.similarity, dtype=torch.float32,
                                                    device=query.device))
            self.trace.update(tokens_per_position=query.shape[0],
                              key_tokens=recent.shape[1] + query.shape[0] * max(map(len, self.history_groups)),
                              history_weights=[float(weight) if group else 0.
                                               for weight, group in zip(weights.detach().tolist(),
                                                                        self.history_groups)])
            mixed = local
            for weight, group in zip(weights, self.history_groups):
                if group:
                    expanded = torch.cat([recent] + [kv[layer_idx] for kv in group], dim=1)
                    mixed = mixed + weight.to(local.dtype) * (self.attend(query, expanded) - local)
            if self.prefix is not None:
                with torch.autocast(device_type=query.device.type, enabled=False):
                    if self.prefix.calibration is None:
                        self.prefix.collect(current, self.current.index)
                        if self.prefix.calibration is not None:
                            self.trace["prefix_calibration"] = self.prefix.calibration
                    elif self.prefix.weights is not None:
                        mixed = self.prefix.read(query, mixed, self.current.index)
                        self.trace["prefix_fast_weights_applied"] = True
            if self.history_fast is not None:
                with torch.autocast(device_type=query.device.type, enabled=False):
                    if self.history_fast.weights is not None:
                        mixed = self.history_fast.read(query, mixed, self.current.index)
                        self.trace["history_fast_weights_applied"] = True
                    update = self.history_fast.update(self.history_fast_entries, layer_idx)
                    if update is not None:
                        self.trace["history_fast_weights_update"] = update
            return mixed
        if self.selected:
            current = torch.cat([kv[layer_idx] for kv in self.selected] + [current], dim=1)
        if layer_idx == 0:
            self.trace.update(tokens_per_position=query.shape[0], key_tokens=current.shape[1])
        return self.attend(query, current)

    def _archive(self, frame):
        if len(self.summaries) < SUMMARIES:
            self.summaries.append(VisualSummary(frame))
        else:
            # 相邻起始帧常很相似，先让每簇获得第二次观测，再开放全部簇竞争。
            candidates = [g for g in self.summaries if g.count == 1] or self.summaries

            def similarity(group):
                norm = max(float(np.linalg.norm(group.center)), 1e-12)
                return float(group.center @ frame.descriptor) / norm, -group.count

            group = max(candidates, key=similarity)
            group.add(frame, self.rope)

        if frame.index < FIXED:
            self.fixed.append(frame)
        elif frame.index in self.admitted:
            self.admitted.remove(frame.index)
            self.anchors.append(frame)
        elif frame.index + self.lookahead > self.current.index:
            self.pending[frame.index] = frame

    def finish(self, decision):
        if self.current is None or self.written_layers != self.layers:
            raise RuntimeError("视觉窗口只提交完成所有跨帧层的输入")
        if decision is not None and decision.candidate >= FIXED:
            pending = self.pending.pop(decision.candidate, None)
            if decision.kept:
                if pending is None:
                    self.admitted.add(decision.candidate)
                else:
                    self.anchors.append(pending)
        self.recent.append(self.current)
        if len(self.recent) >= RECENT:
            self._archive(self.recent.popleft())
        self.current = None
        self.selected = []
        self.next_index += 1
