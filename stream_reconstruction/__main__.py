"""对有序的单目 RGB 图像运行流式重建和锚帧选择。"""

import argparse
from contextlib import closing
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from .abot import reconstruct
from .selection import SelectionConfig
from .gate_data import GateEvidence, load_gate_weights
from .submaps import SubmapWriter, apply_official_loop, write_manifest


def input_paths(source):
    """文件夹按文件名排序，TUM 清单保留严格递增的输入时间。"""
    if source.is_dir():
        paths = sorted(p for p in source.iterdir()
                       if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    else:
        paths = []
        previous = -math.inf
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            timestamp, name = line.split(maxsplit=1)
            timestamp = float(timestamp)
            if not math.isfinite(timestamp) or timestamp <= previous:
                raise ValueError("输入清单的时间戳须有限且严格递增")
            paths.append(source.parent / name)
            previous = timestamp
    if not paths:
        raise ValueError("输入中没有 RGB 图像")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="RGB 目录或 TUM 格式的时间戳清单")
    parser.add_argument("--checkpoint", required=True, type=Path, help="本地 ABot 权重文件或目录")
    parser.add_argument("--output", required=True, type=Path, help="新建的输出目录")
    parser.add_argument("--lookahead", type=int, default=4, help="选锚前观察的后续输入步数")
    parser.add_argument("--capacity", type=int, default=8, help="长期锚帧槽位数")
    parser.add_argument("--min-gap", type=int, default=1, help="两个已接纳锚之间的最小输入间隔")
    parser.add_argument("--visual-context", choices=("structured24", "local12"), default="structured24",
                        help="主干读取 12＋6＋3＋3 个视觉位置，或使用原始 12 帧对照")
    parser.add_argument("--visual-gate-weights", type=Path,
                        help="最后一层视觉注意力的历史融合门 JSON")
    adaptation = parser.add_mutually_exclusive_group()
    adaptation.add_argument("--visual-ttt", action="store_true",
                            help="用历史共视点在线微调视觉注意力门")
    adaptation.add_argument("--prefix-fast-weights", action="store_true",
                            help="用前 12 帧拟合会话快权重，从第 13 帧使用")
    adaptation.add_argument("--history-fast-weights", action="store_true",
                            help="用选中的窗外帧和摘要在线更新快权重")
    parser.add_argument("--revise-depth", action="store_true", help="保存四步几何融合后的锚帧深度")
    parser.add_argument("--pose-mode", choices=("baseline", "anchors", "fixed"), default="baseline",
                        help="选择原始位姿、历史锚，或历史锚加起始三帧")
    parser.add_argument("--pose-solver", choices=("single", "joint"), default="single",
                        help="选择单参考半步融合或多参考联合求解")
    parser.add_argument("--summary", choices=("off", "recent", "full"), default="off",
                        help="联合求解读取三个近期或全历史的方向摘要")
    parser.add_argument("--direction-gain", type=float, default=1.,
                        help="方向残差能量权重，范围为 [0, 1]")
    parser.add_argument("--summary-alignment", choices=("single", "consensus"), default="single",
                        help="摘要写入采用单参考或两个不同子图的旋转共识")
    parser.add_argument("--ttt", choices=("off", "fixed", "online"), default="off",
                        help="关闭门控、使用同协议的固定门或在线训练门")
    gate_options = parser.add_mutually_exclusive_group()
    gate_options.add_argument("--gate-weights", type=Path, help="离线训练生成的融合门 JSON")
    gate_options.add_argument("--capture-gates", action="store_true", help="导出供离线训练使用的几何线性系统")
    parser.add_argument("--profile", action="store_true", help="记录首发、端到端延迟及显存")
    parser.add_argument("--submaps", action="store_true", help="每 10 帧储存子图局部点")
    parser.add_argument("--loop-salad-checkpoint", type=Path, help="官方回环所需的本地 SALAD 权重")
    parser.add_argument("--loop-dino-checkpoint", type=Path, help="官方回环所需的本地 DINO 权重")
    args = parser.parse_args()
    if args.visual_gate_weights and args.visual_context != "structured24":
        parser.error("视觉门权重需要 --visual-context structured24")
    if args.visual_ttt and not args.visual_gate_weights:
        parser.error("视觉 TTT 需要 --visual-gate-weights 初始化")
    if args.prefix_fast_weights and not args.visual_gate_weights:
        parser.error("前缀快权重需要 --visual-gate-weights")
    if args.history_fast_weights and not args.visual_gate_weights:
        parser.error("历史快权重需要 --visual-gate-weights")
    visual_gate = None
    if args.visual_gate_weights:
        from .visual_gate import VisualHistoryGate

        try:
            visual_gate = VisualHistoryGate.load(args.visual_gate_weights)
        except (OSError, ValueError) as error:
            parser.error(f"视觉门加载失败：{error}")
    config = SelectionConfig(lookahead=args.lookahead, capacity=args.capacity, min_gap=args.min_gap)
    if args.revise_depth and args.lookahead != 4:
        parser.error("四步深度融合需要 --lookahead 4")
    if args.pose_mode == "baseline" and args.pose_solver == "joint":
        parser.error("联合求解需要 --pose-mode anchors 或 fixed")
    if args.summary != "off" and (args.pose_mode != "fixed" or args.pose_solver != "joint"):
        parser.error("方向摘要需要 --pose-mode fixed --pose-solver joint")
    if not 0 <= args.direction_gain <= 1:
        parser.error("--direction-gain 须在 [0, 1] 内")
    if args.ttt != "off" and (args.summary != "full" or not 0 < args.direction_gain < 1):
        parser.error("门控实验需要全历史摘要及 (0, 1) 内的初始方向权重")
    if (args.gate_weights or args.capture_gates) and args.ttt != "fixed":
        parser.error("融合门加载和采集需要 --ttt fixed")
    gate_parameters = None
    if args.gate_weights:
        try:
            gate_parameters = load_gate_weights(args.gate_weights)
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"融合门加载失败：{error}")
    training_evidence = GateEvidence() if args.capture_gates else None
    loop_paths = (args.loop_salad_checkpoint, args.loop_dino_checkpoint)
    if any(loop_paths) and not all(path is not None and path.is_file() for path in loop_paths):
        parser.error("回环需要同时提供有效的本地 SALAD 和 DINO 权重")
    paths = input_paths(args.source)
    from abot_recon import ABotRecon

    wrapper = ABotRecon.from_pretrained(
        str(args.checkpoint), local_files_only=True, device="cuda",
        attention_backend="sdpa", loop_closure=False, max_frames=max(22_000, len(paths)))
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    if visual_gate is not None:
        visual_gate.to(next(wrapper.model.network.parameters()).device)
    if args.visual_ttt:
        wrapper.model.network.requires_grad_(False)

    maps = SubmapWriter(args.output / "submaps") if args.submaps else None
    raw_poses, publish_ms, step_ms = [], [], []
    peak_anchor_bytes = peak_summary_bytes = 0
    if args.profile:
        torch.cuda.reset_peak_memory_stats()
    log = args.output / "decisions.jsonl.part"
    trajectory = args.output / "trajectory.jsonl.part"
    steps = reconstruct(wrapper.model.network.eval(), paths, config, revise_depth=args.revise_depth,
                        pose_mode=args.pose_mode, pose_solver=args.pose_solver, summary=args.summary,
                        direction_gain=args.direction_gain, summary_alignment=args.summary_alignment,
                        ttt=args.ttt, profile=args.profile, gate_parameters=gate_parameters,
                        training_evidence=training_evidence, visual_context=args.visual_context,
                        visual_gate=visual_gate, visual_ttt=args.visual_ttt,
                        prefix_fast_weights=args.prefix_fast_weights,
                        history_fast_weights=args.history_fast_weights)
    with (closing(steps), log.open("x", encoding="utf-8") as stream,
          trajectory.open("x", encoding="utf-8") as poses):
        started = perf_counter()
        accepted = 0
        visual_accepted = 0
        for step in steps:
            if step.prediction is not None:
                if args.profile:
                    # 窗外摘要在 GPU 上异步更新，计时须包含本步提交的工作。
                    torch.cuda.synchronize()
                publish_ms.append((perf_counter() - started) * 1000)
                raw = step.prediction["camera_poses"][0, 0].detach().float().cpu().tolist()
                entry = dict(index=step.index, image=paths[step.index].name, raw=raw,
                             pose=step.published_pose.tolist(), mode=args.pose_mode,
                             visual_mode=args.visual_context,
                             solver=args.pose_solver, summary=args.summary,
                             direction_gain=args.direction_gain, summary_alignment=args.summary_alignment,
                             **step.pose_evidence)
                poses.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                accepted += bool(entry.get("ttt", {}).get("accepted", False))
                visual_accepted += bool(entry.get("visual_ttt", {}).get("accepted", False))
                peak_anchor_bytes = max(peak_anchor_bytes, entry.get("anchor_index_bytes", 0))
                peak_summary_bytes = max(peak_summary_bytes, entry.get("summary_index_bytes", 0))
                if args.loop_salad_checkpoint is not None:
                    raw_poses.append(raw)
                if maps is not None:
                    maps.add(step.index, step.published_pose, step.prediction)
            if step.decision is not None:
                decision = step.decision
                if step.revised_depth is not None:
                    np.save(args.output / f"depth_{decision.candidate:06d}.npy", step.revised_depth)
                record = dict(**asdict(decision), image=paths[decision.candidate].name)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            ended = perf_counter()
            if step.prediction is not None:
                step_ms.append((ended - started) * 1000)
            started = ended
    if training_evidence is not None:
        training_evidence.save(args.output / "gate_evidence.npz", len(paths), args.direction_gain)
    trajectory.rename(args.output / "trajectory.jsonl")
    log.rename(args.output / "decisions.jsonl")
    if maps is not None:
        maps.finish()
    if args.profile:
        write_manifest(args.output / "profile.json", dict(
            frames=len(publish_ms), visual_mode=args.visual_context,
            first_pose_p95_ms=float(np.percentile(publish_ms, 95)),
            end_to_end_p95_ms=float(np.percentile(step_ms, 95)),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), accepted_ttt_updates=accepted,
            accepted_visual_ttt_updates=visual_accepted,
            peak_anchor_index_bytes=peak_anchor_bytes, peak_summary_index_bytes=peak_summary_bytes))
    if args.loop_salad_checkpoint is not None:
        apply_official_loop(paths, raw_poses, wrapper, args.output, args.loop_salad_checkpoint,
                            args.loop_dino_checkpoint, None if maps is None else maps.records)


if __name__ == "__main__":
    main()
