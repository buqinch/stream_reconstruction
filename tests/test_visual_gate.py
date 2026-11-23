import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from stream_reconstruction.selection import SelectionConfig
from stream_reconstruction.train_visual_gate import LONG_HORIZONS, balanced_scores, compose_pose, motion_loss
from stream_reconstruction.visual_context import VisualContext
from stream_reconstruction.visual_gate import VisualHistoryGate
from test_visual_context import feed, memory


class VisualGateTests(unittest.TestCase):
    def test_recent_only_has_same_output_before_history(self):
        context = memory()
        gate = VisualHistoryGate()
        tuned = VisualContext(context.layers, context.rope, SelectionConfig(), visual_gate=gate)
        keys = torch.randn(3, 2, 8)
        for candidate in (context, tuned):
            candidate.begin(np.ones(3))
            for layer in range(candidate.layers):
                candidate.append_frame(layer, keys, 2 * keys)
        expected = context.compute_attention(1, keys)
        torch.testing.assert_close(tuned.compute_attention(1, keys), expected, rtol=0, atol=0)

    def test_gate_gradients_use_only_final_attention_layer(self):
        context = memory()
        gate = VisualHistoryGate()
        context.visual_gate = gate
        for index in range(25):
            feed(context, index)
        context.begin(np.eye(3)[0])
        keys = torch.randn(3, 2, 8)
        for layer in range(context.layers):
            context.append_frame(layer, keys, keys)
        self.assertFalse(context.compute_attention(0, keys).requires_grad)
        context.compute_attention(1, keys).sum().backward()
        self.assertTrue(gate.parameters_by_group.grad.isfinite().all())
        self.assertTrue((gate.parameters_by_group.grad[:, 0].abs() > 0).all())

    def test_motion_objective_backpropagates_to_current_pose(self):
        pose = torch.eye(4)
        pose[0, 3] = .2
        pose.requires_grad_()
        targets = np.repeat(np.eye(4)[None], 3, axis=0)
        loss = motion_loss(pose, [torch.eye(4)] * 16, targets)
        loss.backward()
        self.assertGreater(loss.item(), 0.)
        self.assertGreater(pose.grad[0, 3].item(), 0.)

    def test_long_motion_objective_backpropagates_to_two_steps_in_block(self):
        first = torch.eye(4)
        first[0, 3] = .1
        first.requires_grad_()
        second = torch.eye(4)
        second[0, 3] = .2
        second.requires_grad_()
        pose = compose_pose(second, first, first)
        targets = np.full((len(LONG_HORIZONS), 4, 4), np.nan)
        targets[-1] = np.eye(4)
        loss = motion_loss(pose, [torch.eye(4)] * 64, targets, LONG_HORIZONS)
        loss.backward()
        self.assertGreater(first.grad[0, 3].item(), 0.)
        self.assertGreater(second.grad[0, 3].item(), 0.)

    def test_balanced_selection_accounts_for_rotation_and_translation(self):
        metrics = {
            "local12": dict(ate_m=.2, suffix_m=.5, rpe_long=dict(rotation_deg=5.)),
            "low_ate": dict(ate_m=.15, suffix_m=.6, rpe_long=dict(rotation_deg=6.)),
            "balanced": dict(ate_m=.18, suffix_m=.47, rpe_long=dict(rotation_deg=4.8)),
        }
        scores = balanced_scores(metrics)
        self.assertEqual(scores["local12"], 1.)
        self.assertLess(scores["balanced"], scores["low_ate"])

    def test_weight_loader_validates_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text('{"version": 1, "parameters": [[-4, 1], [-4, 2], [-4, 3]]}')
            trained = VisualHistoryGate.load(path)
            self.assertFalse(trained.parameters_by_group.requires_grad)
            path.write_text('{"version": 1, "parameters": [[-4, 0]]}')
            with self.assertRaises(ValueError):
                VisualHistoryGate.load(path)
