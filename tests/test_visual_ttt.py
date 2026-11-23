import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from stream_reconstruction.visual_gate import VisualHistoryGate
from stream_reconstruction.visual_ttt import VisualTTT


class VisualTTTTests(unittest.TestCase):
    def setUp(self):
        self.gate = VisualHistoryGate()
        self.adapter = VisualTTT(self.gate)
        self.adapter.observations[0] = object()
        self.frame = SimpleNamespace(index=24)
        self.base = np.eye(4)
        points = np.array([[.1, .2, 1.], [-.2, .3, 1.], [.2, -.1, 1.]])
        rays = points - [.05, 0., 0.]
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        self.samples = points, rays, np.full(3, 1 / 3.)

    def predict(self, sign=1.):
        pose = torch.eye(4)
        pose[0, 3] = sign * .10 * (self.gate.parameters_by_group[0, 0] + 4.)
        return pose

    def update(self, sign=1.):
        match = SimpleNamespace(query=np.arange(30))
        with (patch("stream_reconstruction.visual_ttt.pose_evidence", return_value=match),
              patch("stream_reconstruction.visual_ttt.partition", return_value=[self.samples] * 2)):
            return self.adapter.update(self.frame, self.base, self.predict(),
                                       [0], lambda: self.predict(sign))

    def test_accepts_update_when_validation_improves(self):
        info = self.update()
        self.assertTrue(info["accepted"])
        self.assertLess(info["loss_after"], info["loss_before"])
        self.assertAlmostEqual(float(self.gate.parameters_by_group[0, 0]), -3.9, places=5)

    def test_restores_gate_when_validation_worsens(self):
        info = self.update(sign=-1.)
        self.assertFalse(info["accepted"])
        torch.testing.assert_close(self.gate.parameters_by_group, self.adapter.initial)

    def test_excludes_references_inside_recent_window(self):
        self.adapter.observations[20] = object()
        with patch("stream_reconstruction.visual_ttt.pose_evidence") as match:
            info = self.adapter.update(self.frame, self.base, self.predict(), [20], self.predict)
        self.assertFalse(info["accepted"])
        match.assert_not_called()
