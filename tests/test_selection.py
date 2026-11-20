import unittest
from dataclasses import replace

import cv2
import numpy as np

from stream_reconstruction.selection import AnchorSelector, Observation, SelectionConfig, observe


def fixture(index, scene=1, angle=0, descriptor=None):
    rng = np.random.default_rng(scene)
    features = rng.integers(0, 256, (24, 32), dtype=np.uint8)
    theta = np.deg2rad(angle)
    rays = np.tile([np.sin(theta), 0, np.cos(theta)], (24, 1)).astype(np.float32)
    if descriptor is None:
        descriptor = [1, 0] if scene == 1 else [0, 1]
    return Observation(index, np.array(descriptor, np.float32), np.arange(24), features, rays)


class SelectionTests(unittest.TestCase):
    def test_repeated_views_and_tail(self):
        selector = AnchorSelector(SelectionConfig(lookahead=2))
        output = [selector.push(fixture(i)) for i in range(5)]
        self.assertEqual(output[:2], [None, None])
        self.assertTrue(output[2].kept)
        self.assertEqual((output[2].candidate, output[2].decided_at), (0, 2))
        self.assertFalse(output[3].kept)
        self.assertEqual(output[4].reason, "已有锚覆盖该视角")
        self.assertEqual([r.candidate for r in selector.finish()], [3, 4])
        self.assertEqual(selector.finish(), [])
        with self.assertRaises(ValueError):
            selector.push(fixture(5))

    def test_alternative_view_then_revisit(self):
        selector = AnchorSelector(SelectionConfig(lookahead=1))
        frames = [fixture(i, angle=v) for i, v in enumerate([0, 0, 10, 10, 0, 0])]
        output = [selector.push(f) for f in frames][1:]
        self.assertEqual([d.candidate for d in output if d.kept], [0, 2])
        self.assertEqual(output[2].reason, "有视差的新视角")
        self.assertAlmostEqual(output[2].parallax_deg, 10, places=3)
        self.assertEqual(output[4].reference, 0)

    def test_stable_replacement_is_bounded(self):
        selector = AnchorSelector(SelectionConfig(lookahead=1, capacity=1))
        output = [selector.push(fixture(i, scene=s)) for i, s in enumerate([1, 1, 2, 2])]
        self.assertTrue(output[1].kept)
        self.assertEqual(output[2].reason, "缺少后续共视支持")
        self.assertEqual(output[3].reason, "稳定的新区域")
        self.assertEqual(output[3].evicted, 0)
        self.assertEqual(len(selector.anchors), 1)
        self.assertEqual(len(selector.pending), 1)

    def test_spatially_concentrated_and_transient_features(self):
        selector = AnchorSelector(SelectionConfig(lookahead=1))
        selector.push(replace(fixture(0), cells=np.zeros(24, np.int32)))
        self.assertEqual(selector.push(fixture(1, 2)).reason, "可靠点覆盖不足")
        self.assertEqual(selector.push(fixture(2, 1)).reason, "缺少后续共视支持")

    def test_reject_invalid_inputs_before_advancing(self):
        selector = AnchorSelector()
        with self.assertRaises(ValueError):
            selector.push(fixture(1))
        selector.push(fixture(0))
        with self.assertRaises(ValueError):
            selector.push(fixture(1, descriptor=[1, 0, 0]))
        selector.push(fixture(1))
        with self.assertRaises(ValueError):
            SelectionConfig(capacity=0)
        with self.assertRaises(ValueError):
            SelectionConfig(min_overlap=float("nan"))

    def test_pure_rotation_and_sampling(self):
        gray = np.random.default_rng(3).integers(0, 256, (140, 252), dtype=np.uint8)
        detector = cv2.ORB_create(nfeatures=160)
        pose = np.eye(4)
        angle = np.deg2rad(18)
        pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]

        def sample(y, x):
            return np.tile([0, 0, 2], (len(x), 1)), np.ones(len(x))

        def rotated(y, x):
            xyz, conf = sample(y, x)
            return xyz @ pose[:3, :3], conf

        a = observe(0, [2., 0], gray, np.eye(4), detector, sample)
        b = observe(1, [2., 0], gray, pose, detector, rotated)
        self.assertGreater(len(a.features), 8)
        np.testing.assert_allclose(a.descriptor, [1, 0])
        np.testing.assert_allclose(a.rays, b.rays, atol=1e-6)
        with self.assertRaises(ValueError):
            observe(2, [0., 0], gray, pose, detector, rotated)

    def test_configurable_confidence_floor(self):
        gray = np.random.default_rng(6).integers(0, 256, (140, 252), dtype=np.uint8)
        detector = cv2.ORB_create(nfeatures=64)

        def sample(y, x):
            return np.tile([0., 0., 1.], (len(x), 1)), np.full(len(x), 0.2)

        self.assertGreater(len(observe(0, [1.], gray, np.eye(4), detector, sample).cells), 0)
        self.assertEqual(len(observe(0, [1.], gray, np.eye(4), detector, sample, 0.5).cells), 0)


if __name__ == "__main__":
    unittest.main()
