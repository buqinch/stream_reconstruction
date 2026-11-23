"""以小尺寸随机权重调用 ABot 原生逐帧解码器、相邻位姿头和点图头。"""

from copy import copy
from functools import partial
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from abot_recon.modeling.pi3.models.layers.adjacent_pose_head import AdjacentPoseHead
from abot_recon.modeling.pi3.models.layers.attention import FlashAttentionRope
from abot_recon.modeling.pi3.models.layers.block import BlockRope
from abot_recon.modeling.pi3.models.layers.pos_embed import PositionGetter, RoPE2D
from abot_recon.modeling.pi3.models.layers.transformer_head import LinearPts3d, TransformerDecoder
from abot_recon.modeling.streaming.attention import StreamingFlashAttention
from abot_recon.modeling.streaming.network import ABotReconNetwork
from abot_recon.modeling.streaming.rope3d import RoPE3D
from stream_reconstruction.abot import reconstruct
from stream_reconstruction.selection import Decision, SelectionConfig
from stream_reconstruction.visual_context import stream_network


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 32, 14, stride=14)
        self.calls = 0

    def forward(self, x, is_training=True):
        self.calls += 1
        return {"x_norm_patchtokens": self.patch(x).flatten(2).transpose(1, 2)}


def small_network():
    # 完整发布模型约十亿参数，夹具仅缩小通道、层数和输入网格。
    net = ABotReconNetwork.__new__(ABotReconNetwork)
    nn.Module.__init__(net)
    net.encoder = PatchEncoder()
    net.image_mean, net.image_std = 0., 1.
    net.patch_size, net.patch_start_idx = 14, 5
    net.register_token = nn.Parameter(torch.randn(1, 1, 5, 32))
    net.rope3d_embed = RoPE3D(8, (1, 14, 14), 128, fhw_dim=(4, 2, 2))
    net.global_pos_encoding, net.pos_type = "rope3d", "rope100"
    net.position_getter, net.rope = PositionGetter(), RoPE2D(freq=100)
    net.local_window_frames, net.num_reference_frames = 12, 0
    net.streaming_num_summary_tokens, net.paged_max_summary_frames = 0, 0
    net.memory_mode, net.num_dec_blk_not_to_checkpoint = "streaming", 4
    global_attention = partial(StreamingFlashAttention, global_pos_encoding="rope3d",
                               rope3d_embed=net.rope3d_embed, local_window_frames=12)
    net.decoder = nn.ModuleList([
        BlockRope(32, 4, rope=net.rope, qk_norm=True, qkv_bias=True, attn_class=attention)
        for attention in (FlashAttentionRope, global_attention) * 2])
    net.use_global_points, net.train_conf, net.point_z_log_max = False, True, 10.
    head_decoder = partial(TransformerDecoder, 64, 32, dec_embed_dim=32, depth=1,
                           dec_num_heads=4, rope=net.rope)
    net.point_decoder, net.camera_decoder, net.conf_decoder = head_decoder(), head_decoder(), head_decoder()
    net.point_head, net.conf_head = LinearPts3d(14, 32, 3), LinearPts3d(14, 32, 1)
    net.camera_pose_mode, net.relative_camera_head_type = "relative_adjacent", "token_pair"
    net.camera_head = AdjacentPoseHead(32, 32, rot_correction_kernel=3, init_std=0.05)
    return net.eval()


class BackboneTests(unittest.TestCase):
    @torch.inference_mode()
    def test_each_history_group_reaches_raw_heads(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21)
            net, history = stream_network(small_network(), SelectionConfig(), "structured24")
            images = torch.rand(41, 1, 1, 3, 28, 28)
        state = None
        for t in range(40):
            out = net._forward_frame_paged(images[t], frame_idx=t, ref_hidden=None, camera_state=state)
            state = out["camera_state"]
            history.finish(Decision(t - 4, t, True, "受控接纳") if t >= 4 else None)

        def predict(drop=None):
            # 各次探测从同一前缀、相同相机状态起步，仅改变本步读到的那组 K/V。
            ctx = copy(history)
            net.encoder.context = net._paged_manager = ctx
            begin = ctx.begin

            def select(descriptor):
                begin(descriptor)
                self.assertEqual(ctx.trace["positions"], 24)
                sizes = [len(ctx.trace["fixed"]), len(ctx.trace["anchors"]),
                         len(ctx.trace["recent"]) - 1, len(ctx.trace["summaries"])]
                if drop is not None:
                    start = sum(sizes[:drop])
                    del ctx.selected[start:start + sizes[drop]]

            with patch.object(ctx, "begin", side_effect=select):
                return net._forward_frame_paged(images[40], frame_idx=40, ref_hidden=None,
                                                camera_state=state)

        full = predict()
        for group in (0, 1, 3):
            partial_result = predict(drop=group)
            for key in ("camera_poses", "local_points"):
                self.assertGreater((full[key] - partial_result[key]).abs().max().item(), 1e-7,
                                   (group, key))

    def test_default_window_changes_both_heads_after_warmup(self):
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            net = small_network()
            inputs = torch.rand(16, 3, 28, 28)

        def read(index):
            image = inputs[index]
            return image, (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        def run(**options):
            return [step for step in reconstruct(net, range(len(inputs)), **options)
                    if step.prediction is not None]

        with patch("stream_reconstruction.abot._input_image", side_effect=read):
            baseline, structured = run(visual_context="local12"), run()
            restart = reconstruct(net, [0])
            try:
                first = next(restart)
            finally:
                restart.close()
        self.assertEqual(net.encoder.calls, 2 * len(inputs) + 1)
        torch.testing.assert_close(first.prediction["local_points"], structured[0].prediction["local_points"])
        for key in ("camera_poses", "local_points", "conf"):
            for old, new in zip(baseline[:12], structured[:12]):
                torch.testing.assert_close(old.prediction[key], new.prediction[key], rtol=2e-5, atol=3e-6)
            diff = (structured[12].prediction[key] - baseline[12].prediction[key]).abs().max().item()
            self.assertGreater(diff, 1e-5, key)
        trace = structured[12].pose_evidence["visual_context"]
        self.assertEqual((trace["fixed"], trace["summaries"], trace["positions"]), ([0], [], 13))
