"""用带参考位姿的场景离线训练点与方向融合门。"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .submaps import write_manifest
from .trajectory import REGULARIZER
from .ttt import OnlineWeights, solve_update


def metric_scale(raw, reference):
    valid = np.isfinite(reference[:120]).all(axis=(1, 2))
    x = raw[:120, :3, 3][valid]
    y = reference[:120, :3, 3][valid]
    if len(x) < 3:
        raise ValueError("起始 120 帧有效参考位姿不足")
    x, y = x - x.mean(0), y - y.mean(0)
    left, values, right = np.linalg.svd(y.T @ x)
    energy = np.square(x).sum()
    positive = values[0] > 0 and values[1] > values[0] * 1e-5
    if not np.isfinite(energy) or energy <= 1e-12 or not positive:
        raise ValueError("起始 120 帧缺少可估计尺度的运动")
    scale = (values[0] + values[1] + np.linalg.det(left @ right) * values[2]) / energy
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("轨迹尺度无效")
    return float(scale)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scene(directory, item, initial_direction):
    evidence = directory / item["evidence"]
    reference_path = directory / item["reference"]
    with np.load(reference_path, allow_pickle=False) as source:
        reference = source["poses"].astype(np.float64)
    with np.load(evidence, allow_pickle=False) as data:
        count = int(data["frame_count"])
        if int(data["version"]) != 1 or float(data["direction_gain"]) != initial_direction:
            raise ValueError("缓存版本或采集初始权重与训练配置不一致")
        if reference.shape != (count, 4, 4):
            raise ValueError("参考位姿数量须与完整采集轨迹相同")
        rows = [json.loads(line) for line in
                (evidence.parent / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        if [row["index"] for row in rows] != list(range(count)):
            raise ValueError("采集位姿须完整且连续")
        raw = np.asarray([row["raw"] for row in rows], dtype=np.float64)
        scale = metric_scale(raw, reference)
        indices = data["frame"]
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices <= 0) or np.any(indices >= count)
                or np.any(np.diff(indices) <= 0)):
            raise ValueError("缓存帧号须为有序且有效的相邻帧终点")
        n = len(indices)
        arrays = {key: data[key] for key in ("prior", "scale", "features", "normals", "rhs")}
        shapes = {"prior": (n, 4, 4), "scale": (n,), "features": (n, 2, 3),
                  "normals": (n, 2, 6, 6), "rhs": (n, 2, 6)}
        if any(arrays[key].shape != shape or not np.isfinite(arrays[key]).all()
               for key, shape in shapes.items()) or np.any(arrays["scale"] <= 0):
            raise ValueError("缓存的线性系统形状或数值无效")
        valid = np.isfinite(reference).all(axis=(1, 2))
        keep = valid[indices] & valid[indices - 1]
        indices = indices[keep]
        if not len(indices):
            raise ValueError("没有与真实相邻位姿匹配的证据")
        target = np.linalg.solve(reference[indices - 1], reference[indices])
        target[:, :3, 3] /= scale
        update = np.linalg.solve(target, arrays["prior"][keep])
        rotations = np.stack([cv2.Rodrigues(matrix)[0].ravel() for matrix in update[:, :3, :3]])
        translation = update[:, :3, 3] / arrays["scale"][keep, None]
        values = [arrays[key][keep] for key in ("features", "normals", "rhs")]
        values.append(np.concatenate((rotations, translation), axis=1))

    metadata = dict(scene=item["scene"], samples=len(indices), metric_scale=scale,
                    evidence_sha256=digest(evidence), reference_sha256=digest(reference_path))
    return tuple(torch.as_tensor(value, dtype=torch.float64) for value in values), metadata


def pose_loss(parameters, scene):
    features, normals, rhs, targets = scene
    gains = (parameters * features).sum(-1).sigmoid()
    regularizer = torch.as_tensor(REGULARIZER, dtype=parameters.dtype)
    estimated = solve_update(gains, normals, rhs, regularizer)
    sigma = torch.as_tensor([np.deg2rad(1)] * 3 + [.015] * 3, dtype=parameters.dtype)
    return torch.nn.functional.smooth_l1_loss(estimated / sigma, targets / sigma)


def mean_loss(parameters, scenes):
    return torch.stack([pose_loss(parameters, scene) for scene in scenes]).mean()


def train(training, validation, initial, epochs):
    parameters = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([parameters], lr=.03)
    baseline = float(mean_loss(initial, validation))
    best_loss = baseline
    best, best_epoch = initial.clone(), 0
    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = mean_loss(parameters, training)
        objective = loss + 1e-4 * (parameters - initial).square().mean()
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            parameters.clamp_(-8., 8.)
            selected = float(mean_loss(parameters, validation))
            if not np.isfinite(selected):
                raise ValueError("验证损失为非有限值")
            if selected < best_loss:
                best, best_loss, best_epoch = parameters.clone(), selected, epoch
            history.append(dict(epoch=epoch, train_loss=float(loss.detach()),
                                validation_loss=selected))
    return best, dict(best_epoch=best_epoch, validation_initial=baseline,
                      validation_best=best_loss, epochs=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="训练和验证场景 JSON")
    parser.add_argument("--output", type=Path, required=True, help="新建权重和训练记录目录")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("训练轮数必须为正整数")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_items, validation_items = manifest["train"], manifest["validation"]
    scene_names = [item["scene"] for item in (*train_items, *validation_items)]
    if not train_items or not validation_items or len(set(scene_names)) != len(scene_names):
        parser.error("训练与验证须包含不同且不重复的场景")
    initial_direction = float(manifest.get("direction_gain", .1))
    if not 0 < initial_direction < 1:
        parser.error("初始方向能量权重须在 (0, 1) 内")
    datasets, metadata = {}, {}
    for name, items in (("train", train_items), ("validation", validation_items)):
        pairs = [read_scene(args.manifest.parent, item, initial_direction) for item in items]
        datasets[name], metadata[name] = zip(*pairs)
    for key in ("evidence_sha256", "reference_sha256"):
        fingerprints = [entry[key] for group in metadata.values() for entry in group]
        if len(set(fingerprints)) != len(fingerprints):
            parser.error("训练与验证数据存在相同的证据或参考轨迹")
    initial = torch.tensor(OnlineWeights(initial_direction, online=False).initial)
    best, history = train(datasets["train"], datasets["validation"], initial, args.epochs)
    args.output.mkdir(parents=True, exist_ok=False)
    write_manifest(args.output / "history.json", dict(scenes=metadata, **history))
    write_manifest(args.output / "gates.json", dict(version=1, parameters=best.tolist(),
                                                    best_epoch=history["best_epoch"],
                                                    training_scenes=metadata))


if __name__ == "__main__":
    main()
