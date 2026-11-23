import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from stream_reconstruction.submaps import SubmapWriter, relocate_submaps


class SubmapTests(unittest.TestCase):
    def test_relocation_preserves_local_points_and_first_release(self):
        with TemporaryDirectory() as scratch:
            path = Path(scratch) / 'submaps'
            writer = SubmapWriter(path)
            prediction = {
                'local_points': torch.tensor([[[[[0., 0., 1.]]]]]),
                'conf': torch.ones((1, 1, 1, 1, 1)),
            }
            first = np.eye(4)
            first[0, 3] = 0.5
            writer.add(0, first, prediction)
            writer.finish()
            filename = path / writer.records[0]['points']
            before = np.eye(4)[None]
            after = before.copy()
            after[0, 0, 3] = 2
            entries = relocate_submaps(writer.records, before, after)
            self.assertEqual(entries[0]['world_from_submap'][0][3], 2.5)
            self.assertEqual(json.loads((path/'submaps.json').read_text())[0]['world_from_submap'][0][3], .5)
            np.testing.assert_array_equal(np.load(filename), [[0., 0., 1.]])
