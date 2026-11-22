"""使用已经到达的观测决定整帧是否进入锚帧缓存。"""

from collections import deque
from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    lookahead: int = 4
    capacity: int = 8
    min_gap: int = 1
    min_quality: float = 0.25
    min_overlap: float = 0.45
    min_parallax_deg: float = 2.0
    min_confidence: float = 0.0

    def __post_init__(self):
        for value in (self.lookahead, self.capacity, self.min_gap):
            if type(value) is not int or value < 1:
                raise ValueError("观察步数、锚帧容量和最小间隔必须为正整数")
        if not (0 < self.min_quality <= 1 and 0 < self.min_overlap <= 1):
            raise ValueError("覆盖率阈值必须在 (0, 1] 内")
        if not (0 < self.min_parallax_deg < 180):
            raise ValueError("视差阈值必须在 (0, 180) 度内")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("点置信度阈值必须在 [0, 1] 内")


@dataclass(frozen=True)
class Observation:
    index: int
    descriptor: np.ndarray
    cells: np.ndarray
    features: np.ndarray
    rays: np.ndarray

    local_points: np.ndarray | None = None

    @property
    def quality(self):
        return len(np.unique(self.cells)) / 24


def observe(index, descriptor, gray, pose, detector, sample_geometry, min_confidence=0.0):
    """从当前帧提取有限个局部描述子，射线统一到世界坐标系。"""
    descriptor = np.asarray(descriptor, np.float32)
    norm = np.linalg.norm(descriptor)
    if descriptor.ndim != 1 or not isfinite(norm) or norm <= 0:
        raise ValueError("帧描述子须为非零有限向量")
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 1:
        raise ValueError("选帧需要非空的八位灰度图")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("相机到世界的变换须为有限的 4×4 矩阵")
    keypoints, features = detector.detectAndCompute(gray, None)
    if features is None:
        return Observation(index, descriptor / norm, np.empty(0, np.int32),
                           np.empty((0, 32), np.uint8), np.empty((0, 3), np.float32),
                           np.empty((0, 3), np.float32))
    xy = np.rint([k.pt for k in keypoints]).astype(np.int32)
    x, y = xy.T
    xyz, confidence = sample_geometry(y, x)
    if xyz.shape != (len(x), 3) or confidence.shape != (len(x),):
        raise ValueError("每个采样位置需要一个局部三维点和一个置信度")
    reliable = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    reliable &= np.isfinite(confidence) & (confidence >= min_confidence)
    x, y, xyz = x[reliable], y[reliable], xyz[reliable].astype(np.float64)
    cells = (y * 4 // gray.shape[0]) * 6 + (x * 6 // gray.shape[1])
    rays = xyz @ pose[:3, :3].T
    lengths = np.linalg.norm(rays, axis=1, keepdims=True)
    if np.any(lengths <= 0):
        raise ValueError("相机旋转产生了无效射线")
    return Observation(index, descriptor / norm, cells, features[reliable],
                       (rays / lengths).astype(np.float32), xyz.astype(np.float32))


def mutual_matches(candidate, reference, matcher):
    if min(len(candidate.features), len(reference.features)) < 8:
        return []
    # 比率检验排除重复纹理，反向最近邻剔除多点争抢同一对应。
    pairs = matcher.knnMatch(candidate.features, reference.features, k=2)
    reverse = {m.queryIdx: m.trainIdx
               for m in matcher.match(reference.features, candidate.features)}
    return [a for a, b in pairs
            if a.distance < 0.75 * b.distance and reverse.get(a.trainIdx) == a.queryIdx]


def compare(candidate, reference, matcher):
    """覆盖率以候选帧的有效网格为分母，视差扣除了相机旋转。"""
    good = mutual_matches(candidate, reference, matcher)
    if len(good) < 8:
        return 0.0, 0.0
    query = [m.queryIdx for m in good]
    train = [m.trainIdx for m in good]
    count = len(np.unique(candidate.cells[query]))
    if count < 3 or len(np.unique(reference.cells[train])) < 3:
        return 0.0, 0.0
    coverage = count / len(np.unique(candidate.cells))
    cosines = (candidate.rays[query] * reference.rays[train]).sum(axis=1)
    angles = np.rad2deg(np.arccos(np.clip(cosines, -1, 1)))
    return coverage, float(np.median(angles))


@dataclass(frozen=True)
class Decision:
    candidate: int
    decided_at: int
    kept: bool
    reason: str
    support: int | None = None
    reference: int | None = None
    coverage: float | None = None
    parallax_deg: float | None = None
    evicted: int | None = None


class AnchorSelector:
    def __init__(self, config=SelectionConfig()):
        self.config = config
        self.pending = deque()
        self.anchors = deque()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.next_index = 0
        self.dimension = None
        self.closed = False

    def push(self, frame):
        """按输入序号推进一步，到期时返回一个可追踪的选择决定。"""
        if self.closed or frame.index != self.next_index:
            raise ValueError("输入须从零连续编号，结束后请新建选择器")
        if self.dimension is not None and frame.descriptor.shape != self.dimension:
            raise ValueError("同一序列的描述子维度必须一致")
        self.pending.append(frame)
        self.dimension = frame.descriptor.shape
        self.next_index += 1
        if len(self.pending) <= self.config.lookahead:
            return None
        candidate = self.pending.popleft()
        return self._decide(candidate, frame.index)

    def _decide(self, frame, now):
        config = self.config
        if frame.quality < config.min_quality:
            return Decision(frame.index, now, False, "可靠点覆盖不足")
        if self.anchors and frame.index - self.anchors[-1].index < config.min_gap:
            return Decision(frame.index, now, False, "与上一锚间隔过短")
        support = None
        # 同时验证紧邻输入与最近到达输入，每个目标至多计算两次近邻匹配。
        neighbors = [self.pending[0]]
        if self.pending[-1].index != neighbors[0].index:
            neighbors.append(self.pending[-1])
        for item in neighbors:
            coverage, _ = compare(frame, item, self.matcher)
            if item.quality >= config.min_quality and coverage >= config.min_overlap:
                support = item.index
                break
        if support is None:
            return Decision(frame.index, now, False, "缺少后续共视支持")

        # 对最相似的两个锚做局部匹配。任一锚已充分覆盖此视角，就避免重复保存。
        nearest = sorted(self.anchors, key=lambda a: float(a.descriptor @ frame.descriptor),
                         reverse=True)[:2]
        best_ref, best_coverage, best_angle = None, 0.0, None
        for anchor in nearest:
            coverage, angle = compare(frame, anchor, self.matcher)
            if coverage >= config.min_overlap and angle < config.min_parallax_deg:
                return Decision(frame.index, now, False, "已有锚覆盖该视角", support,
                                anchor.index, coverage, angle)
            if best_ref is None or coverage > best_coverage:
                best_ref, best_coverage, best_angle = anchor.index, coverage, angle
        if not nearest:
            reason = "首个稳定锚"
        elif best_coverage < config.min_overlap:
            reason = "稳定的新区域"
        else:
            reason = "有视差的新视角"
        evicted = None
        if len(self.anchors) == config.capacity:
            evicted = self.anchors.popleft().index
        self.anchors.append(frame)
        return Decision(frame.index, now, True, reason, support, best_ref, best_coverage,
                        best_angle, evicted)

    def finish(self):
        """流结束时记录尾帧的实际状态，保持固定的决策时限。"""
        self.closed = True
        tail = [Decision(f.index, self.next_index - 1, False, "序列结尾观察不足")
                for f in self.pending]
        self.pending.clear()
        return tail
