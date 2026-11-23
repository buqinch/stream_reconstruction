import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from stream_reconstruction.selection import Decision, SelectionConfig
from stream_reconstruction.visual_context import VisualContext, stream_network
from stream_reconstruction.visual_replay import VisualReplay
from test_visual_backbone import small_network
from test_visual_context import descriptor, memory


class FourAnchorTests(unittest.TestCase):
    def run_cache(self, context, decisions, length):
        traces = []
        for index in range(length):
            context.begin(descriptor(index))
            values = torch.full((3, 2, 8), float(index))
            context.append_frame(0, values, values)
            context.compute_attention(0, torch.zeros_like(values))
            context.finish(decisions.get(index))
            traces.append(context.trace)
        return traces

    def test_equal_recent_and_summary_replay_use_the_c_schedule(self):
        config = SelectionConfig()
        rope = memory().rope
        decisions = {index: Decision(index - 4, index, index != 12, "受控选择")
                     for index in range(4, 101)}
        chosen = self.run_cache(VisualContext(1, rope, config, mode="anchors4"), decisions, 101)
        schedule = VisualReplay(tuple(tuple(row["anchors"]) for row in chosen), decisions)
        matched = self.run_cache(VisualContext(1, rope, config, mode="matched-recent", replay=schedule),
                                 {}, 101)
        summaries = self.run_cache(VisualContext(1, rope, config, mode="anchors4-summary", replay=schedule),
                                   {}, 101)
        self.assertEqual(chosen[12]["anchors"], [0])
        self.assertEqual(chosen[13]["anchors"], [1, 0])
        self.assertIn(2, chosen[14]["anchors"])
        self.assertEqual(chosen[100]["recent"], list(range(89, 101)))
        self.assertEqual(sum(len(r["anchors"]) for r in chosen[:12]), 0)
        self.assertEqual([i for i, r in enumerate(summaries) if r.get("summary_archive") == 8], [19])
        self.assertFalse(decisions[12].kept)
        for index, (c, b, d) in enumerate(zip(chosen, matched, summaries)):
            self.assertEqual(c["anchors"], d["anchors"])
            self.assertEqual(c["library"], d["library"])
            self.assertTrue(all(anchor <= index - 12 for anchor in c["anchors"]))
            self.assertEqual(len(b["recent"]), len(c["recent"]) + len(c["anchors"]))
            self.assertEqual(b["key_tokens"], c["key_tokens"])
            self.assertEqual(d["key_tokens"], c["key_tokens"] + 3 * len(d["summaries"]))
            self.assertFalse(c["fixed"])
            self.assertTrue(all(group["last"] <= index - 12 for group in d["summaries"]))

    def test_replay_rejects_current_frame_or_different_image(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [Path("frame.jpg")]
            trail = dict(index=0, image="frame.jpg", visual_mode="anchors4",
                         visual_context=dict(anchors=[0]))
            (root / "decisions.jsonl").write_text(
                json.dumps(dict(candidate=0, decided_at=0, kept=False, image="frame.jpg")) + "\n")
            (root / "trajectory.jsonl").write_text(json.dumps(trail) + "\n")
            with self.assertRaisesRegex(ValueError, "因果"):
                VisualReplay.load(root, paths, 4)
            trail["visual_context"]["anchors"] = []
            (root / "trajectory.jsonl").write_text(json.dumps(trail) + "\n")
            with self.assertRaisesRegex(ValueError, "输入帧"):
                VisualReplay.load(root, [Path("other.jpg")], 4)
            self.assertEqual(VisualReplay.load(root, paths, 4).anchors, ((),))

    @torch.inference_mode()
    def test_native_prediction_reads_current_anchors_and_summary(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20)
            model = small_network()
            inputs = torch.rand(28, 1, 1, 3, 28, 28)
        network, context = stream_network(model, SelectionConfig(), "anchors4-summary")
        previous = None
        for index in range(27):
            result = network._forward_frame_paged(inputs[index], frame_idx=index,
                                                  ref_hidden=None, camera_state=previous)
            previous = result["camera_state"]
            context.finish(Decision(index - 4, index, True, "受控选择") if index >= 4 else None)

        def predict(drop=None):
            from copy import copy

            copied = copy(context)
            network.encoder.context = network._paged_manager = copied
            begin = copied.begin

            def select(value):
                begin(value)
                if drop == "anchor":
                    del copied.selected[0]
                elif drop == "summary":
                    copied.selected.pop()

            with patch.object(copied, "begin", side_effect=select):
                return network._forward_frame_paged(inputs[27], frame_idx=27,
                                                    ref_hidden=None, camera_state=previous)

        complete = predict()
        for kind in ("anchor", "summary"):
            changed = predict(kind)
            self.assertGreater((changed["camera_poses"] - complete["camera_poses"]).abs().max().item(), 1e-6)
            self.assertGreater((changed["local_points"] - complete["local_points"]).abs().max().item(), 1e-6)
        self.assertEqual(model.encoder.calls, 30)
