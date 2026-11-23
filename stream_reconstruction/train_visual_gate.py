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


def read_sequence(directory, item):
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
    targets = np.full((len(paths), len(HORIZONS), 4, 4), np.nan)
    for column, gap in enumerate(HORIZONS):
        starts = reference[:-gap]
        ends = reference[gap:]
        valid = (np.isfinite(starts).all(axis=(1, 2)) & np.isfinite(ends).all(axis=(1, 2)))
        relative = np.linalg.solve(starts[valid], ends[valid])
        relative[:, :3, 3] /= scale
        targets[np.flatnonzero(valid) + gap, column] = relative
    return paths, targets


def motion_loss(pose, previous, targets):
    pairs = [(previous[gap - 1], targets[column], gap)
             for column, gap in enumerate(HORIZONS)
             if len(previous) >= gap and np.isfinite(targets[column]).all()]
    if not pairs:
        return None
    last, truth, gaps = zip(*pairs)
    predicted = torch.linalg.solve(torch.stack(last), pose.expand(len(pairs), -1, -1))
    target = torch.as_tensor(np.asarray(truth), dtype=pose.dtype, device=pose.device)
    sqrt_gap = torch.tensor(gaps, dtype=pose.dtype, device=pose.device).sqrt()[:, None, None]
    difference = (predicted[:, :3] - target[:, :3]) / sqrt_gap
    rotation = torch.nn.functional.smooth_l1_loss(difference[:, :, :3] / .02,
                                                  torch.zeros_like(difference[:, :, :3]))
    displacement = torch.nn.functional.smooth_l1_loss(difference[:, :, 3] / .03,
                                                      torch.zeros_like(difference[:, :, 3]))
    return rotation + displacement


def train_scene(model, gate, optimizer, paths, targets, stride):
    previous = deque(maxlen=max(HORIZONS))
    losses = []
    with closing(reconstruct(model, paths, visual_gate=gate, gradient_interval=stride)) as steps:
        for step in steps:
            if step.prediction is None:
                continue
            pose = step.prediction["camera_poses"][0, 0].float()
            if pose.requires_grad:
                loss = motion_loss(pose, previous, targets[step.index])
                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    with torch.no_grad():
                        gate.parameters_by_group.clamp_(-12., 12.)
                    losses.append(float(loss.detach()))
            previous.appendleft(pose.detach())
    if not losses:
        raise ValueError("场景缺少可用于微调的样本")
    return dict(samples=len(losses), loss=float(np.mean(losses)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="只包含训练场景的清单")
    parser.add_argument("--checkpoint", type=Path, required=True, help="本地 ABot 权重")
    parser.add_argument("--output", type=Path, required=True, help="新建微调输出目录")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--stride", type=int, default=4, help="每隔多少帧反传一次")
    args = parser.parse_args()
    if min(args.epochs, args.stride) < 1:
        parser.error("训练轮数和反传间隔须为正整数")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest["train"]
    names = [item["scene"] for item in items]
    if not names or len(names) != len(set(names)):
        parser.error("训练场景须存在且互不重复")
    datasets = [(item["scene"], *read_sequence(args.manifest.parent, item)) for item in items]

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
                score = train_scene(model, gate, optimizer, paths, targets, args.stride)
                scores[name] = score
                print(f"第 {epoch} 轮 {name} {score}", flush=True)
            history.append(dict(epoch=epoch, scenes=scores))
            write_manifest(args.output / "history.json", history)
        write_manifest(args.output / f"epoch-{epoch}.json", dict(
            version=1, epoch=epoch, parameters=gate.parameters_by_group.detach().cpu().tolist(),
            training_scenes=names))


if __name__ == "__main__":
    main()
