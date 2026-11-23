import unittest
from types import SimpleNamespace

import numpy as np

from stream_reconstruction.trajectory import PoseEvidence, REGULARIZER
from stream_reconstruction.ttt import OnlineWeights, split_tracks


class TTTTests(unittest.TestCase):
    def test_same_current_feature_and_history_track_share_one_group(self):
        frame = SimpleNamespace(features=np.empty(4), index=8)
        left = PoseEvidence(1, np.arange(3), np.ones((3, 3)), np.eye(4), np.arange(3))
        right = PoseEvidence(4, np.array([0, 2, 3]), np.ones((3, 3)), np.eye(4), np.arange(3))
        references = {
            1: SimpleNamespace(folds=np.array([1, 0, 1])),
            4: SimpleNamespace(folds=np.array([1, 2, 2])),
        }
        folds, groups = split_tracks(frame, [left, right], references, recent={4})
        np.testing.assert_array_equal(folds, [1, 0, -2, 2])
        self.assertEqual([m.query.tolist() for m in groups[0]], [[1]])
        self.assertEqual([m.query.tolist() for m in groups[1]], [[0], [0]])
        self.assertEqual([m.query.tolist() for m in groups[2]], [[3]])

    def test_gradient_affects_only_available_gate_and_is_bounded(self):
        weights = OnlineWeights(0.1, online=True)
        features = np.ones((2, 3))
        points = np.array([[-.1, -.1, 1.], [.1, 0., 1.], [0., .1, 1.]])
        rays = points - np.array([.008, 0., 0.])
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        normal = np.diag([0., 0., 0., 1e4, 0., 0.])
        rhs = np.array([0., 0., 0., 100., 0., 0.])
        updated = weights._gradient_step(
            features, [normal, np.zeros((6, 6))], [rhs, np.zeros(6)], REGULARIZER,
            (points, rays, np.ones(3) / 3))
        self.assertIsNotNone(updated)
        np.testing.assert_array_equal(updated[1], weights.initial[1])
        self.assertGreater(np.linalg.norm(updated[0] - weights.initial[0]), 0)
        self.assertLessEqual(np.max(np.abs(updated - weights.initial)), .02000001)
        np.testing.assert_array_equal(weights.parameters, weights.initial)
