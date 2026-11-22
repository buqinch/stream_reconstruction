import unittest

import cv2
import numpy as np

from stream_reconstruction.directions import Axis, BLOCK_FRAMES, DirectionMemory, plane_axes
from stream_reconstruction.trajectory import joint_step


def axis(x=0., y=0., z=1.):
    return Axis(np.array([x, y, z]), 1., 1.)


class DirectionTests(unittest.TestCase):
    def test_planar_tiles_ignore_invalid_depth_and_low_confidence(self):
        y, x = np.mgrid[:35, :63]
        points = np.stack((x / 100, y / 100, np.full(x.shape, 2.)), axis=2)
        confidence = np.ones_like(x)
        points[:9, :, 2] = np.nan
        axes = plane_axes(points, confidence, 0.25)
        self.assertEqual(len(axes), 1)
        self.assertAlmostEqual(abs(axes[0].normal[2]), 1.)
        self.assertEqual(plane_axes(points, np.zeros_like(x), 0.25), [])

    def test_sign_invariance_and_replacement(self):
        memory = DirectionMemory('full')
        memory.commit(0, [axis()], np.eye(3))
        self.assertEqual(memory.read([axis()], np.eye(3)), [])
        memory.commit(1, [axis(z=-1.)], np.eye(3))
        result = memory.read([axis()], np.eye(3))
        self.assertEqual(result[0].support, 2)
        self.assertAlmostEqual(result[0].scatter, 0.)
        rotation = cv2.Rodrigues(np.array([0.8, 0., 0.]))[0]
        memory.commit(1, [axis()], rotation)
        self.assertEqual(memory.read([axis()], np.eye(3)), [])
        memory.commit(1, [axis()], np.eye(3))
        self.assertEqual(memory.read([axis()], np.eye(3))[0].support, 2)

    def test_commit_boundary_and_recent_expiration(self):
        full, recent = (DirectionMemory(mode) for mode in ('full', 'recent'))
        for frame in range(4 * BLOCK_FRAMES):
            for memory in (full, recent):
                if frame < 2 * BLOCK_FRAMES - 1:
                    self.assertEqual(memory.read([axis()], np.eye(3)), [])
                rotation = np.eye(3) if frame < 2 * BLOCK_FRAMES else None
                memory.after_publish(frame, [axis()], rotation)
        self.assertEqual(full.read([axis()], np.eye(3))[0].support, 2)
        self.assertEqual(recent.read([axis()], np.eye(3)), [])
        self.assertEqual(set(recent.contributions), {1})

    def test_rotation_constraint_without_points_has_no_translation(self):
        memory = DirectionMemory('full')
        for block in range(2):
            memory.commit(block, [axis()], np.eye(3))
        prior = np.eye(4)
        prior[:3, :3] = cv2.Rodrigues(np.array([0.12, 0., 0.]))[0]
        prior[:3, 3] = [1., 2., 3.]
        matches = memory.read([axis()], prior[:3, :3])
        count, pose = joint_step(prior, None, [], matches)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(pose[:3, 3], prior[:3, 3])
        self.assertLess(np.linalg.norm(pose[:3, :3] - np.eye(3)),
                        np.linalg.norm(prior[:3, :3] - np.eye(3)))

    def test_second_nonparallel_direction_observes_twist(self):
        memory = DirectionMemory('full')
        axes = [axis(), axis(x=1., z=0.)]
        for block in range(2):
            memory.commit(block, axes, np.eye(3))
        prior = np.eye(4)
        prior[:3, :3] = cv2.Rodrigues(np.array([0., 0., 0.12]))[0]
        one_axis = memory.read(axes[:1], prior[:3, :3])
        self.assertIsNone(joint_step(prior, None, [], one_axis))
        two_axes = memory.read(axes, prior[:3, :3])
        self.assertEqual({match.slot for match in two_axes}, {0, 1})
        _, pose = joint_step(prior, None, [], two_axes)
        self.assertLess(np.linalg.norm(pose[:3, :3] - np.eye(3)),
                        np.linalg.norm(prior[:3, :3] - np.eye(3)))
