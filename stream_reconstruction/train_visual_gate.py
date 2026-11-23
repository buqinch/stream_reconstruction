"""冻结 ABot，使用带参考位姿的流式序列训练最后一层注意力门。"""

import argparse
from collections import deque
from contextlib import closing
import json
from pathlib import Path

import numpy as np
import torch

from .abot import reconstruct
from .submaps import write_manifest
from .train_gates import metric_scale
from .visual_gate import VisualHistoryGate


HORIZONS = (1, 8, 16)
LONG_HORIZONS = (*HORIZONS, 32, 64)


def balanced_scores(metrics, baseline="local12"):
    """在同一验证场景用基线归一化三个误差后取等权平均。"""
    values = {name: np.array([row["ate_m"], row["suffix_m"], row["rpe_long"]["rotation_deg"]])
              for name, row in metrics.items()}
    if any(not np.isfinite(row).all() or (row < 0).any() for row in values.values()):
        raise ValueError("选模指标须为非负有限值")
    reference = values[baseline]
    if (reference <= 0).any():
        raise ValueError("选模基线的三个误差须为正")
    return {name: float((row / reference).mean()) for name, row in values.items()}


def read_sequence(directory, item, horizons=HORIZONS):
    image_dir = directory / item["rgb"]
    paths = sorted(path for path in image_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".png", ".jpeg"})[:item["frames"]]
    if len(paths) != item["frames"] or len(paths) < 120:
        raise ValueError("训练图像须与清单数量一致并至少包含 120 帧")
    with np.load(directory / item["reference"], allow_pickle=False) as archive:
        reference = archive["poses"][:len(paths)].copy()
    if reference.shape != (len(paths), 4, 4):
        raise ValueError("参考位姿数量与图像不一致")
    with (directory / item["baseline"]).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for _, line in zip(paths[:120], stream)]
    if [row["index"] for row in rows] != list(range(120)):
        raise ValueError("基线轨迹缺少完整的起始帧")
    raw = np.array([row["raw"] for row in rows])
    scale = metric_scale(raw, reference)
    targets = np.full((len(paths), len(horizons), 4, 4), np.nan)
    for column, gap in enumerate(horizons):
        starts = reference[:-gap]
        ends = reference[gap:]
        valid = (np.isfinite(starts).all(axis=(1, 2)) & np.isfinite(ends).all(axis=(1, 2)))
        relative = np.linalg.solve(starts[valid], ends[valid])
        relative[:, :3, 3] /= scale
        targets[np.flatnonzero(valid) + gap, column] = relative
    return paths, targets


def motion_loss(pose, previous, targets, horizons=HORIZONS):
    pairs = [(previous[gap - 1], targets[column], gap)
             for column, gap in enumerate(horizons)
             if len(previous) >= gap and np.isfinite(targets[column]).all()]
    if not pairs:
        return None
    last, truth, gaps = zip(*pairs)
    predicted = torch.linalg.solve(torch.stack(last), pose.expand(len(pairs), -1, -1))
    target = torch.as_tensor(np.asarray(truth), dtype=pose.dtype, device=pose.device)
    spans = torch.tensor(gaps, dtype=pose.dtype, device=pose.device)
    sqrt_gap = spans.sqrt()[:, None, None]
    difference = (predicted[:, :3] - target[:, :3]) / sqrt_gap
    rotation = torch.nn.functional.smooth_l1_loss(difference[:, :, :3] / .02,
                                                  torch.zeros_like(difference[:, :, :3]),
                                                  reduction="none").mean(dim=(1, 2))
    displacement = torch.nn.functional.smooth_l1_loss(difference[:, :, 3] / .03,
                                                      torch.zeros_like(difference[:, :, 3]),
                                                      reduction="none").mean(dim=1)
    long = (spans >= 32).to(pose.dtype)
    rotation_weight, translation_weight = 1 + long, 1 + .5 * long
    return ((rotation @ rotation_weight) / rotation_weight.sum()
            + (displacement @ translation_weight) / translation_weight.sum())


def compose_pose(pose, native_previous, composed_previous):
    # 从上游已修正的相邻增量重建块内的梯度路径。
    return composed_previous @ torch.linalg.solve(native_previous.detach(), pose)


def train_scene(model, gate, optimizer, paths, targets, stride, horizons=HORIZONS, block_size=1):
    previous = deque(maxlen=max(horizons))
    losses = []
    pending = []
    native_previous = None
    updates = 0
    with closing(reconstruct(model, paths, visual_gate=gate, gradient_interval=stride)) as steps:
        for step in steps:
            if step.prediction is None:
                continue
            pose = step.prediction["camera_poses"][0, 0].float()
            composed = pose
            if block_size > 1 and step.index % block_size:
                composed = compose_pose(pose, native_previous, previous[0])
            if pose.requires_grad:
                loss = motion_loss(composed, previous, targets[step.index], horizons)
                if loss is not None:
                    pending.append(loss)
                    losses.append(float(loss.detach()))
            previous.appendleft(composed)
            native_previous = pose.detach()
            if (step.index + 1) % block_size == 0 or step.index == len(paths) - 1:
                if pending:
                    optimizer.zero_grad()
                    torch.stack(pending).mean().backward()
                    torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    with torch.no_grad():
                        gate.parameters_by_group.clamp_(-12., 12.)
                    pending.clear()
                    updates += 1
                previous = deque((item.detach() for item in previous), maxlen=previous.maxlen)
    if not losses:
        raise ValueError("场景缺少可用于微调的样本")
    return dict(samples=len(losses), updates=updates, loss=float(np.mean(losses)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="只包含训练场景的清单")
    parser.add_argument("--checkpoint", type=Path, required=True, help="本地 ABot 权重")
    parser.add_argument("--output", type=Path, required=True, help="新建微调输出目录")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--stride", type=int, default=4, help="每隔多少帧反传一次")
    parser.add_argument("--long-horizon", action="store_true",
                        help="增加 32 和 64 帧目标，并对块内的位姿组合反传")
    parser.add_argument("--block-size", type=int, default=8, help="长程模式的反传块长度")
    args = parser.parse_args()
    if min(args.epochs, args.stride, args.block_size) < 1:
        parser.error("训练轮数、反传间隔和块长度须为正整数")
    horizons = LONG_HORIZONS if args.long_horizon else HORIZONS
    block_size = args.block_size if args.long_horizon else 1
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest["train"]
    names = [item["scene"] for item in items]
    if not names or len(names) != len(set(names)):
        parser.error("训练场景须存在且互不重复")
    datasets = [(item["scene"], *read_sequence(args.manifest.parent, item, horizons)) for item in items]

    from abot_recon import ABotRecon

    wrapper = ABotRecon.from_pretrained(str(args.checkpoint), local_files_only=True, device="cuda",
                                       attention_backend="sdpa", loop_closure=False)
    model = wrapper.model.network.eval().requires_grad_(False)
    gate = VisualHistoryGate().cuda()
    optimizer = torch.optim.Adam(gate.parameters(), lr=.045)
    args.output.mkdir(parents=True, exist_ok=False)
    history = []
    for epoch in range(args.epochs + 1):
        if epoch:
            scores = {}
            for name, paths, targets in datasets:
                score = train_scene(model, gate, optimizer, paths, targets, args.stride,
                                    horizons, block_size)
                scores[name] = score
                print(f"第 {epoch} 轮 {name} {score}", flush=True)
            history.append(dict(epoch=epoch, scenes=scores))
            write_manifest(args.output / "history.json", history)
        write_manifest(args.output / f"epoch-{epoch}.json", dict(
            version=1, epoch=epoch, parameters=gate.parameters_by_group.detach().cpu().tolist(),
            training_scenes=names, horizons=horizons, block_size=block_size))


if __name__ == "__main__":
    main()
