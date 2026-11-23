"""检查真实帧去重、摘要时序和跨帧注意力可见性。"""

import unittest

import numpy as np
import torch
from abot_recon.modeling.streaming.rope3d import RoPE3D, apply_rotary_emb

from stream_reconstruction.selection import Decision, SelectionConfig
from stream_reconstruction.visual_context import VisualContext, VisualFrame, VisualSummary


def descriptor(index):
    return np.eye(3, dtype=np.float32)[index % 3]


def memory(lookahead=4, capacity=8):
    rope = RoPE3D(8, (1, 14, 14), 128, fhw_dim=(4, 2, 2))
    return VisualContext(2, rope, SelectionConfig(lookahead=lookahead, capacity=capacity))


def feed(context, index, keep=True):
    context.begin(descriptor(index))
    kv = torch.full((3, 2, 8), float(index))
    for layer in range(context.layers):
        context.append_frame(layer, kv, kv)
    candidate = index - context.lookahead
    decision = Decision(candidate, index, keep, "测试") if candidate >= 0 else None
    context.finish(decision)
    return context.trace


class ContextTests(unittest.TestCase):
    def test_budget_causality_and_retrieval(self):
        context = memory()
        for t in range(80):
            trace = feed(context, t)
            recent, fixed, anchors = (trace[name] for name in ("recent", "fixed", "anchors"))
            ids = recent + fixed + anchors
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(recent, list(range(max(0, t - 11), t + 1)))
            self.assertEqual(fixed, list(range(min(3, max(0, t - 11)))))
            self.assertTrue(all(3 <= k <= t - 12 for k in anchors))
            self.assertTrue(all(g["last"] <= t - 12 for g in trace["summaries"]))
            self.assertLessEqual(trace["positions"], 24)
            if t >= 24:
                self.assertEqual((len(recent), len(anchors), len(fixed), len(trace["summaries"])),
                                 (12, 6, 3, 3))
                self.assertEqual(anchors[0] % 3, t % 3)
                self.assertEqual(sum(g["count"] for g in trace["summaries"]), t - 11)
        self.assertEqual(len(context.anchors), 8)
        self.assertEqual((len(context.recent), len(context.fixed), len(context.pending)), (11, 3, 0))

    def test_delayed_and_rejected_anchors(self):
        context = memory(lookahead=16, capacity=2)
        for t in range(64):
            trace = feed(context, t, keep=t % 2 == 0)
            self.assertTrue(all(k + 16 < t and k % 2 == 0 for k in trace["anchors"]))
            self.assertLessEqual(len(context.pending), 5)
            self.assertLessEqual(len(trace["anchors"]), 2)
        self.assertEqual(len(trace["anchors"]), 2)
        self.assertFalse(context.admitted)

    def test_only_selected_tokens_reach_sdpa(self):
        context = memory()
        for t in range(40):
            feed(context, t)
        context.begin(descriptor(40))
        query = torch.zeros(3, 2, 8)
        context.append_frame(0, query, query)
        expected = sum(kv[0, 1] for kv in context.selected) / 24
        torch.testing.assert_close(context.compute_attention(0, query), expected)
        self.assertEqual((context.trace["tokens_per_position"], context.trace["key_tokens"]), (3, 72))
        # Query 为零时所有 token 等权，修改最远固定帧和聚合槽的 V 可精确预测差值。
        previous = context.compute_attention(0, query)
        for position in (0, -1):
            value = context.selected[position][0, 1]
            value.add_(24)
            updated = context.compute_attention(0, query)
            torch.testing.assert_close(updated - previous, torch.ones_like(query))
            previous = updated

    def test_summary_uses_common_phase_and_full_patch_grid(self):
        context = memory()
        rope = context.rope
        base = torch.arange(32, dtype=torch.float32).reshape(1, 2, 2, 8)
        frames = []
        for index in (0, 73):
            phase = rope(1, 1, 1, 1, base.device, f_start=index, f_end=index + 1)
            keys = apply_rotary_emb(base, phase).transpose(1, 2)
            kv = torch.stack((keys, base + index), dim=1)
            frames.append(VisualFrame(index, descriptor(0), kv))
        original = frames[1].kv.clone()
        group = VisualSummary(frames[0])
        group.add(frames[1], rope)
        torch.testing.assert_close(group.kv[:, 0], frames[0].kv[:, 0])
        torch.testing.assert_close(group.kv[:, 1], base + 36.5)
        torch.testing.assert_close(frames[1].kv, original)

    def test_partial_frames_and_position_limit(self):
        context = memory()
        with self.assertRaises(ValueError):
            context.begin(np.zeros(3, dtype=np.float32))
        context.begin(descriptor(0))
        with self.assertRaises(RuntimeError):
            context.begin(descriptor(1))
        with self.assertRaises(RuntimeError):
            context.finish(None)
        self.assertEqual(context.next_index, 0)
        context = memory()
        context.next_index = 128
        with self.assertRaisesRegex(ValueError, "max_frames"):
            context.begin(descriptor(0))
