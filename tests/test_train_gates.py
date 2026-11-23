import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from stream_reconstruction.gate_data import GateEvidence, load_gate_weights
from stream_reconstruction.train_gates import metric_scale, read_scene, train
from stream_reconstruction.trajectory import REGULARIZER
from stream_reconstruction.ttt import OnlineWeights, solve_update


class TrainGatesTests(unittest.TestCase):
    def test_metric_scale_recovers_rotated_and_shifted_reference(self):
        raw = np.repeat(np.eye(4)[None], 12, axis=0)
        positions = np.arange(12)
        raw[:, :3, 3] = np.column_stack((positions, positions % 3, positions % 4))
        reference = raw.copy()
        rotation = cv2.Rodrigues(np.array([.2, -.5, 1.]))[0]
        reference[:, :3, 3] = 2.5 * (raw[:, :3, 3] @ rotation.T) + [1., -7., 3.]
        reference[5] = np.nan
        self.assertAlmostEqual(metric_scale(raw, reference), 2.5)

    def test_training_cache_targets_match_inverse_camera_update(self):
        reference = np.repeat(np.eye(4)[None], 4, axis=0)
        reference[:, :3, 3] = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [1., 1., 1.]]
        raw = reference.copy()
        raw[:, :3, 3] /= 2.
        true_delta = np.linalg.solve(raw[1], raw[2])
        update = np.eye(4)
        update[:3, :3] = cv2.Rodrigues(np.array([.01, -.02, .03]))[0]
        update[:3, 3] = [.08, .02, -.04]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            evidence = GateEvidence()
            evidence.add(2, true_delta @ update, 2., np.ones((2, 3)),
                         np.zeros((2, 6, 6)), np.zeros((2, 6)))
            evidence.save(directory / "gate_evidence.npz", len(raw), .1)
            np.savez(directory / "reference.npz", poses=reference)
            with (directory / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
                for index, pose in enumerate(raw):
                    stream.write(json.dumps(dict(index=index, raw=pose.tolist())) + "\n")
            item = dict(scene="场景", evidence="gate_evidence.npz", reference="reference.npz")
            scene, metadata = read_scene(directory, item, .1)
            self.assertEqual(metadata["samples"], 1)
            np.testing.assert_allclose(scene[-1][0], [.01, -.02, .03, .04, .01, -.02])

    def test_training_reduces_validation_loss_and_serialized_gains_are_finite(self):
        features = torch.tensor([[[1., .6, .1], [1., .2, .4]],
                                 [[1., .3, .5], [1., .5, .2]]], dtype=torch.float64)
        normals = torch.eye(6, dtype=features.dtype).repeat(2, 2, 1, 1) * 100.
        rhs = torch.zeros((2, 2, 6), dtype=features.dtype)
        rhs[:, 0, 3] = 10.
        rhs[:, 1, 2] = 10.
        regularizer = torch.tensor(REGULARIZER)
        gains = torch.tensor([[.3, .5], [.3, .5]], dtype=features.dtype)
        target = solve_update(gains, normals, rhs, regularizer)
        training = features[:1], normals[:1], rhs[:1], target[:1]
        validation = features[1:], normals[1:], rhs[1:], target[1:]
        initial = torch.tensor(OnlineWeights(.1, online=False).initial)
        parameters, history = train([training], [validation], initial, 80)
        self.assertGreater(history["best_epoch"], 0)
        self.assertLess(history["validation_best"], history["validation_initial"] * .5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gates.json"
            path.write_text(json.dumps(dict(version=1, parameters=parameters.tolist())), encoding="utf-8")
            np.testing.assert_array_equal(load_gate_weights(path), parameters.numpy())
            path.write_text(json.dumps(dict(version=1, parameters=[[0., 0., 0.]])), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_gate_weights(path)
