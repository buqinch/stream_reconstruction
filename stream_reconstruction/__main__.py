"""对有序的单目 RGB 图像运行流式重建和锚帧选择。"""

import argparse
from contextlib import closing
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .abot import reconstruct
from .selection import SelectionConfig


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
    args = parser.parse_args()
    config = SelectionConfig(lookahead=args.lookahead, capacity=args.capacity, min_gap=args.min_gap)
    if args.revise_depth and args.lookahead != 4:
        parser.error("四步深度融合需要 --lookahead 4")
    if args.pose_mode == "baseline" and args.pose_solver == "joint":
        parser.error("联合求解需要 --pose-mode anchors 或 fixed")
    if args.summary != "off" and (args.pose_mode != "fixed" or args.pose_solver != "joint"):
        parser.error("方向摘要需要 --pose-mode fixed --pose-solver joint")
    if not 0 <= args.direction_gain <= 1:
        parser.error("--direction-gain 须在 [0, 1] 内")
    paths = input_paths(args.source)
    from abot_recon import ABotRecon

    wrapper = ABotRecon.from_pretrained(
        str(args.checkpoint), local_files_only=True, device="cuda",
        attention_backend="sdpa", loop_closure=False)
    args.output.mkdir(parents=True, exist_ok=False)
    log = args.output / "decisions.jsonl.part"
    trajectory = args.output / "trajectory.jsonl.part"
    steps = reconstruct(wrapper.model.network.eval(), paths, config, revise_depth=args.revise_depth,
                        pose_mode=args.pose_mode, pose_solver=args.pose_solver, summary=args.summary,
                        direction_gain=args.direction_gain, summary_alignment=args.summary_alignment)
    with (closing(steps), log.open("x", encoding="utf-8") as stream,
          trajectory.open("x", encoding="utf-8") as poses):
        for step in steps:
            if step.prediction is not None:
                raw = step.prediction["camera_poses"][0, 0].detach().float().cpu().tolist()
                entry = dict(index=step.index, image=paths[step.index].name, raw=raw,
                             pose=step.published_pose.tolist(), mode=args.pose_mode,
                             solver=args.pose_solver, summary=args.summary,
                             direction_gain=args.direction_gain, summary_alignment=args.summary_alignment,
                             **step.pose_evidence)
                poses.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            if step.decision is None:
                continue
            decision = step.decision
            if step.revised_depth is not None:
                np.save(args.output / f"depth_{decision.candidate:06d}.npy", step.revised_depth)
            record = dict(**asdict(decision), image=paths[decision.candidate].name)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    trajectory.rename(args.output / "trajectory.jsonl")
    log.rename(args.output / "decisions.jsonl")


if __name__ == "__main__":
    main()
