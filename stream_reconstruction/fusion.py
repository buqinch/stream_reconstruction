"""利用已经到达的四个视角修订旧帧深度，输入仅含预测和灰度图。"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    gray: np.ndarray
    points: np.ndarray
    confidence: np.ndarray
    pose: np.ndarray


DELAY = 4
SAMPLE_STRIDE = 4
TILE_PIXELS = 56
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))


def sample(image, xy):
    coords = (xy / SAMPLE_STRIDE).astype(np.float32).reshape(-1, 1, 2)
    return cv2.remap(image, coords, None, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[:, 0]


def source_depth_in_target(points, target_pose, source_pose):
    relative = np.linalg.solve(target_pose, source_pose)
    return points @ relative[2, :3] + relative[2, 3]


def revise(window):
    if len(window) != DELAY + 1 or any(
        frame.index != window[0].index + offset for offset, frame in enumerate(window)
    ):
        raise ValueError("修订需要目标帧及连续到达的四个后续输入")
    anchor = window[0]
    original = anchor.points[:, :, 2]
    h, w = original.shape
    info = dict(points=0, observations=0, tiles=0, changed_pixels=0)
    corners = cv2.goodFeaturesToTrack(anchor.gray, maxCorners=64, qualityLevel=.01,
                                      minDistance=7, blockSize=7)
    if corners is None:
        return original, info
    start = corners[:, 0].copy()
    bounds = np.array([(w - 1) * SAMPLE_STRIDE, (h - 1) * SAMPLE_STRIDE])
    ids = np.flatnonzero((start <= bounds).all(axis=1))
    old_z = sample(original, start)
    old_conf = sample(anchor.confidence, start)
    logs = np.zeros(len(start), np.float64)
    weights = np.zeros_like(logs)
    hits = np.zeros(len(start), np.int16)
    xy = start[ids]
    previous = anchor.gray
    for source in window[1:]:
        if not len(ids):
            break
        forward, ok_f, _ = cv2.calcOpticalFlowPyrLK(previous, source.gray, xy, None, **LK)
        if forward is None:
            break
        backward, ok_b, _ = cv2.calcOpticalFlowPyrLK(source.gray, previous, forward, None, **LK)
        if backward is None:
            break
        survives = (ok_f.ravel() > 0) & (ok_b.ravel() > 0)
        survives &= np.linalg.norm(backward - xy, axis=1) <= 1
        survives &= np.isfinite(forward).all(axis=1) & (forward >= 0).all(axis=1)
        survives &= (forward <= bounds).all(axis=1)
        ids, xy, previous = ids[survives], forward[survives], source.gray
        if not len(ids):
            break
        new_z = source_depth_in_target(sample(source.points, xy), anchor.pose, source.pose)
        ratio = new_z / old_z[ids]
        weight = old_conf[ids] * sample(source.confidence, xy)
        valid = (ratio >= .8) & (ratio <= 1.25) & np.isfinite(weight) & (weight > 0)
        chosen = ids[valid]
        weights[chosen] += weight[valid]
        logs[chosen] += weight[valid] * np.log(ratio[valid])
        hits[chosen] += 1
    info.update(points=int((hits > 0).sum()), observations=int(hits.sum()))
    if info["points"] < 8 or info["observations"] < 32:
        return original, info
    valid = hits > 0
    tile = (start[valid] // TILE_PIXELS).astype(np.int32)
    tile_step = TILE_PIXELS // SAMPLE_STRIDE
    ny, nx = (h + tile_step - 1) // tile_step, (w + tile_step - 1) // tile_step
    tile_id = tile[:, 1] * nx + tile[:, 0]
    count = np.bincount(tile_id, minlength=nx * ny)
    # 先按轨迹归一化，避免存活时间长的单点取代多个不同位置的观测。
    track_weight = weights[valid] / hits[valid]
    mass = np.bincount(tile_id, weights=track_weight, minlength=nx * ny)
    summed = np.bincount(tile_id, weights=track_weight * logs[valid] / weights[valid],
                         minlength=nx * ny)
    supported = count >= 2
    correction = np.zeros(nx * ny, dtype=np.float32)
    correction[supported] = np.clip(.5 * summed[supported] / mass[supported],
                                    -np.log(1.1), np.log(1.1))
    grid = correction.reshape(ny, nx)
    residual = grid[np.arange(h)[:, None] // tile_step, np.arange(w)[None, :] // tile_step]
    revised = original * np.exp(residual)
    info.update(tiles=int(supported.sum()), changed_pixels=int((revised != original).sum()))
    return revised, info
