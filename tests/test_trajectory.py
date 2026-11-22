import unittest
from dataclasses import replace

import cv2
import numpy as np

from stream_reconstruction.selection import Decision, Observation
from stream_reconstruction.trajectory import PoseMemory, Reference, estimate_pose


def scene():
    rng = np.random.default_rng(14)
    points = rng.uniform(-1, 1, (96, 3))
    points[:, 2] += 4
    return Observation(0, np.array([1.]), np.arange(96) % 24,
                       rng.integers(0, 256, (96, 32), dtype=np.uint8),
                       points / np.linalg.norm(points, axis=1)[:, None], points)


class TrajectoryTests(unittest.TestCase):
    def test_reference_to_camera_direction_and_damping(self):
        ref = scene()
        offset = np.array([0.08, 0., 0.])
        current = replace(ref, local_points=ref.local_points - offset)
        result = estimate_pose(current, Reference(ref, np.eye(4)),
                               cv2.BFMatcher(cv2.NORM_HAMMING), np.eye(4))
        count, pose = result
        self.assertEqual(count, 96)
        np.testing.assert_allclose(pose[:3, 3], offset / 2, atol=1e-5)

    def test_scale_and_coverage_rejected(self):
        ref = scene()
        for current in (replace(ref, local_points=2 * ref.local_points),
                        replace(ref, cells=np.zeros(96, int))):
            self.assertIsNone(estimate_pose(current, Reference(ref, np.eye(4)),
                                            cv2.BFMatcher(cv2.NORM_HAMMING), np.eye(4)))

    def test_only_older_distinct_references_and_bounded_history(self):
        ref = scene()
        memory = PoseMemory("fixed", 8, 12)
        for i in range(32):
            choice = Decision(i-1, i, True, "测试") if i else None
            _, evidence = memory.advance(replace(ref, index=i), np.eye(4), choice)
            candidates = evidence["candidates"]
            self.assertEqual(len(set(candidates)), len(candidates))
            self.assertTrue(all(k <= i-12 for k in candidates))
            self.assertLessEqual(len(candidates), 9)
        self.assertEqual(candidates, [0, 1, 2])
        self.assertEqual(len(memory.pending), 1)

    def test_delayed_commit_and_prior_feedback(self):
        ref = scene()
        memory = PoseMemory("anchors", 8, 1)
        memory.advance(ref, np.eye(4), None)
        raw = np.eye(4)
        raw[0, 3] = 0.06
        _, log = memory.advance(replace(ref, index=1), raw, Decision(0, 1, True, "测试"))
        self.assertEqual(log["candidates"], [])
        corrected, _ = memory.advance(replace(ref, index=2), raw, Decision(1, 2, False, "测试"))
        self.assertAlmostEqual(corrected[0, 3], 0.03, places=5)
        invalid = replace(ref, index=3, features=ref.features[:0])
        result, _ = memory.advance(invalid, raw, Decision(2, 3, False, "测试"))
        np.testing.assert_allclose(result, corrected)
