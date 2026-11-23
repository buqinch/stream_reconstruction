"""保存子图局部点，回环后只更新子图到世界的变换。"""

import json

import numpy as np

from .directions import BLOCK_FRAMES


class SubmapWriter:
    def __init__(self, directory, *, stride=8):
        self.directory = directory
        self.directory.mkdir()
        self.records = []
        self.points = []
        self.first_pose = None
        self.first_index = 0
        self.stride = stride

    def add(self, index, pose, prediction):
        if index % BLOCK_FRAMES == 0:
            self._flush()
            self.first_pose = pose.copy()
            self.first_index = index
        if index % 10:
            return
        points = prediction["local_points"][0, 0, ::self.stride, ::self.stride].float().cpu().numpy()
        conf = prediction["conf"][0, 0, ::self.stride, ::self.stride, 0].float().sigmoid().cpu().numpy()
        valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0) & (conf >= .25)
        to_submap = np.linalg.solve(self.first_pose, pose)
        self.points.append((points[valid] @ to_submap[:3, :3].T + to_submap[:3, 3]).astype(np.float32))

    def _flush(self):
        if self.first_pose is None:
            return
        name = f"points_{self.first_index:06d}.npy"
        points = np.concatenate(self.points) if self.points else np.empty((0, 3), dtype=np.float32)
        np.save(self.directory / name, points, allow_pickle=False)
        self.records.append(dict(frame=self.first_index, points=name, count=len(points),
                                 world_from_submap=self.first_pose.tolist()))
        self.points.clear()

    def finish(self):
        self._flush()
        write_manifest(self.directory / "submaps.json", self.records)


def write_manifest(path, entries):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(entries, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def relocate_submaps(records, before, after):
    """将主干回环的世界系校正应用至同帧锚定的局部子图。"""
    relocated = []
    for record in records:
        index = record['frame']
        pose = np.asarray(record['world_from_submap'])
        correction = after[index] @ np.linalg.inv(before[index])
        moved = correction @ pose
        relocated.append(dict(record, world_from_submap=moved.tolist()))
    return relocated


def apply_official_loop(paths, raw_poses, wrapper, directory, salad, dino, records=None):
    """运行上游离线回环，并保留可独立读取的因果轨迹和局部点。"""
    import torch
    from abot_recon.loop_closure import refine_trajectory

    config = wrapper.config.override(loop_salad_checkpoint=salad, loop_dino_checkpoint=dino,
                                     loop_output_dir=directory / "loop")
    original = np.asarray(raw_poses, dtype=np.float32)
    with torch.inference_mode():
        corrected = refine_trajectory(paths, torch.from_numpy(original), wrapper.model, config)
    corrected = corrected.detach().float().cpu().numpy()
    if corrected.shape != original.shape or not np.isfinite(corrected).all():
        raise ValueError("官方回环输出的位姿形状或数值不合法")
    np.save(directory / "abot_loop_poses.npy", corrected, allow_pickle=False)
    if records is not None:
        entries = relocate_submaps(records, original, corrected)
        write_manifest(directory / "submaps/submaps_loop.json", entries)
