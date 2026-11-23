import unittest

import torch
from abot_recon.modeling.streaming.rope3d import apply_rotary_emb

from stream_reconstruction.prefix_fast_weights import PrefixFastWeights
from test_visual_context import memory


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
