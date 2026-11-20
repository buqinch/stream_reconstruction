import unittest

import numpy as np

from stream_reconstruction.fusion import Frame, revise, source_depth_in_target


class FusionTests(unittest.TestCase):
    def test_source_translation_has_correct_direction(self):
        target = np.eye(4)
        source = np.eye(4)
        target[2, 3], source[2, 3] = 3, 4
        np.testing.assert_allclose(source_depth_in_target(np.array([[0., 0., 2.]]),
                                                        target, source), [3])

    def make_window(self, later_z=2., blank=False):
        rng = np.random.default_rng(37)
        gray = rng.integers(0, 256, (280, 504), dtype=np.uint8)
        if blank:
            gray.fill(0)
        frames = []
        for index in range(5):
            points = np.zeros((70, 126, 3), np.float32)
            points[:, :, 2] = later_z if index else 2.
            frames.append(Frame(index, gray, points, np.ones((70, 126), np.float32),
                                np.eye(4, dtype=np.float32)))
        return frames

    def test_identical_geometry_stays_fixed(self):
        frames = self.make_window()
        corrected, info = revise(frames)
        np.testing.assert_array_equal(corrected, frames[0].points[:, :, 2])
        self.assertGreater(info["observations"], 32)
        self.assertLessEqual(info["observations"], 256)

    def test_supported_cells_move_towards_new_depth(self):
        frames = self.make_window(2.2)
        corrected, info = revise(frames)
        self.assertGreater(info["changed_pixels"], 0)
        np.testing.assert_allclose(corrected[corrected != 2.], np.sqrt(2 * 2.2), atol=1e-5)
        self.assertTrue(np.any(corrected == 2.))
        self.assertTrue(np.all(frames[0].points[:, :, 2] == 2.))

    def test_missing_tracks_and_depth_outliers_keep_first_result(self):
        for frames in (self.make_window(blank=True), self.make_window(4.)):
            corrected, info = revise(frames)
            self.assertEqual(info["changed_pixels"], 0)
            np.testing.assert_array_equal(corrected, frames[0].points[:, :, 2])

    def test_skipped_input_rejected(self):
        frames = self.make_window()
        frames[3].index += 1
        with self.assertRaises(ValueError):
            revise(frames)


if __name__ == "__main__":
    unittest.main()
