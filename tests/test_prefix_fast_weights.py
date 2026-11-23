import unittest

import torch
from abot_recon.modeling.streaming.rope3d import apply_rotary_emb

from stream_reconstruction.prefix_fast_weights import HistoricalFastWeights, PrefixFastWeights
from stream_reconstruction.selection import Decision, SelectionConfig
from stream_reconstruction.visual_context import VisualContext
from stream_reconstruction.visual_gate import VisualHistoryGate
from test_visual_context import descriptor, memory


class PrefixFastWeightTests(unittest.TestCase):
    def test_reads_prefix_from_later_time_in_shared_coordinates(self):
        rope = memory().rope
        context = PrefixFastWeights(rope)
        tokens = torch.cat((torch.eye(8), -torch.eye(8)))[:, None, :]
        values = tokens * 2.
        for index in range(12):
            rotated = tokens.clone()
            phase = rope.freqs[index, :rope.fhw_dim[0] // 2]
            rotated[..., :rope.fhw_dim[0]] = apply_rotary_emb(rotated[..., :rope.fhw_dim[0]], phase)
            context.collect(torch.stack((rotated, values)), index)
            if index < 11:
                self.assertIsNone(context.weights)
        self.assertTrue(context.calibration["accepted"])
        self.assertLess(context.calibration["fitted_mse"], .01 * context.calibration["baseline_mse"])
        future = tokens.clone()
        future[..., :rope.fhw_dim[0]] = apply_rotary_emb(
            future[..., :rope.fhw_dim[0]], rope.freqs[40, :rope.fhw_dim[0] // 2])
        readout = context.read(future, torch.zeros_like(tokens), 40)
        torch.testing.assert_close(readout, values * context.gain, rtol=.05, atol=1e-6)
        self.assertIsNone(PrefixFastWeights(rope).weights)

    def test_rejects_fit_that_cannot_improve_mean(self):
        context = PrefixFastWeights(memory().rope)
        for index in range(12):
            context.collect(torch.ones(2, 5, 1, 8), index)
        self.assertFalse(context.calibration["accepted"])
        self.assertIsNone(context.weights)
        self.assertEqual(context.samples, [])

    def test_history_reuses_weights_until_the_selected_sources_change(self):
        fast = HistoricalFastWeights(memory().rope)
        cache = torch.ones(1, 2, 5, 1, 8)
        frame = ((0, 0, 0), cache, 0)
        summary = ((1, 0, 2), cache, 1)
        self.assertTrue(fast.update([frame, summary], 0)["accepted"])
        original = fast.weights
        self.assertIsNone(fast.update([summary, frame], 0))
        self.assertIs(fast.weights, original)
        changed_summary = ((1, 0, 3), cache * 2, 1)
        self.assertTrue(fast.update([frame, changed_summary], 0)["accepted"])
        self.assertIsNot(fast.weights, original)

    def test_history_writes_only_window_external_positions_after_read(self):
        context = VisualContext(1, memory().rope, SelectionConfig(),
                                visual_gate=VisualHistoryGate(), history_fast_weights=True)
        with torch.inference_mode():
            for index in range(40):
                context.begin(descriptor(index))
                values = torch.full((3, 2, 8), float(index))
                context.append_frame(0, torch.ones_like(values), values)
                context.compute_attention(0, torch.ones_like(values))
                trace = context.trace
                if index < 12:
                    self.assertNotIn("history_fast_weights_update", trace)
                if index == 12:
                    self.assertEqual(trace["history_fast_weights_update"]["positions"], 1)
                    torch.testing.assert_close(context.history_fast.mean, torch.zeros(2, 1, 8))
                self.assertEqual(trace.get("history_fast_weights_applied", False), index > 12)
                candidate = index - context.lookahead
                decision = Decision(candidate, index, True, "测试") if candidate >= 0 else None
                context.finish(decision)
        self.assertEqual(trace["history_fast_weights_update"]["positions"], 12)
