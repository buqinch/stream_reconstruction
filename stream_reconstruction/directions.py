"""用独立子图的平面法向维护三个固定查询的方向摘要。"""

from dataclasses import dataclass

import numpy as np


BLOCK_FRAMES = 24
SAMPLE_INTERVAL = 4
RECENT_BLOCKS = 3
SLOTS = 3
MATCH_COS = np.cos(np.deg2rad(12))
QUERY_COS = np.cos(np.deg2rad(20))
QUERY_SEPARATION_COS = np.cos(np.deg2rad(30))
MAX_SCATTER = np.sin(np.deg2rad(10))**2
DIRECTION_SIGMA = np.deg2rad(4)


@dataclass(frozen=True)
class Axis:
    normal: np.ndarray
    coverage: float
    confidence: float


@dataclass(frozen=True)
class DirectionMatch:
    slot: int
    prior_axis: np.ndarray
    local_axis: np.ndarray
    support: int
    scatter: float


def cluster_axes(axes, groups, min_groups, group_count):
    """按符号无关的角度聚合，单个观测组只计一次支持。"""
    if not axes:
        return []
    normals = np.stack([axis.normal for axis in axes])
    weights = np.array([axis.coverage * axis.confidence for axis in axes])
    groups = np.asarray(groups)
    compatible = np.abs(normals @ normals.T) >= MATCH_COS
    active = np.ones(len(axes), dtype=bool)
    result = []
    while active.any() and len(result) < SLOTS:
        scores = compatible @ (weights * active)
        seed = np.argmax(np.where(active, scores, -1))
        neighbors = np.flatnonzero(active & compatible[seed])
        active[neighbors] = False
        chosen = []
        for group in np.unique(groups[neighbors]):
            indices = neighbors[groups[neighbors] == group]
            chosen.append(indices[np.argmax(weights[indices])])
        if len(chosen) < min_groups:
            continue
        vectors = normals[chosen]
        covariance = (vectors.T * weights[chosen]) @ vectors / weights[chosen].sum()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if 1 - eigenvalues[-1] <= MAX_SCATTER:
            result.append(Axis(eigenvectors[:, -1],
                               sum(axes[i].coverage for i in chosen) / group_count,
                               float(np.mean([axes[i].confidence for i in chosen]))))
    return result


def plane_axes(points, confidence, confidence_floor):
    """从下采样点图分块拟合平面，拒绝无深度、线状或起伏较大的区域。"""
    axes = []
    for row in range(4):
        for col in range(6):
            y0, y1 = (row * points.shape[0] // 4, (row + 1) * points.shape[0] // 4)
            x0, x1 = (col * points.shape[1] // 6, (col + 1) * points.shape[1] // 6)
            cell = points[y0:y1, x0:x1].reshape(-1, 3)
            scores = confidence[y0:y1, x0:x1].ravel()
            valid = np.isfinite(cell).all(axis=1) & (cell[:, 2] > 0)
            valid &= np.isfinite(scores) & (scores >= confidence_floor)
            if valid.sum() < max(12, len(cell) // 2):
                continue
            cloud = cell[valid].astype(np.float64)
            centered = cloud - cloud.mean(axis=0)
            values, vectors = np.linalg.eigh(centered.T @ centered)
            if values[1] > max(1e-12, 0.05 * values[2]) and values[0] < 0.03 * values[1]:
                axes.append(Axis(vectors[:, 0], 1., float(scores[valid].mean())))
    return cluster_axes(axes, range(len(axes)), 2, 24)


class DirectionMemory:
    def __init__(self, mode):
        if mode not in ('full', 'recent'):
            raise ValueError('方向摘要模式必须为全历史或近期')
        self.horizon = RECENT_BLOCKS if mode == 'recent' else None
        self.queries = []
        self.moments = np.zeros((SLOTS, 3, 3))
        self.weights = np.zeros(SLOTS)
        self.support = np.zeros(SLOTS, dtype=int)
        self.contributions = {}
        self.block_basis = None
        self.samples = []
        self.committed = None
        self.oldest = 0

    def _remove(self, block):
        old = self.contributions.pop(block, None)
        if old is None:
            return
        moments, weights = old
        self.moments -= moments
        self.weights -= weights
        self.support -= weights > 0
        empty = self.support == 0
        self.moments[empty] = 0
        self.weights[empty] = 0

    def commit(self, block, axes, rotation):
        """将子图法向转到世界系，同编号重对齐时替换该子图的贡献。"""
        if block < self.oldest:
            return
        moments = np.zeros_like(self.moments)
        weights = np.zeros_like(self.weights)
        for axis in sorted(axes, key=lambda item: item.coverage * item.confidence, reverse=True):
            vector = rotation @ axis.normal
            vector /= np.linalg.norm(vector)
            similarity = np.abs(np.asarray(self.queries) @ vector) if self.queries else np.empty(0)
            best = float(similarity.max(initial=0))
            if len(self.queries) < SLOTS and best < QUERY_SEPARATION_COS:
                self.queries.append(vector)
                slot = len(self.queries) - 1
            elif best >= QUERY_COS:
                slot = int(np.argmax(similarity))
            else:
                continue
            weight = axis.coverage * axis.confidence
            if weight > weights[slot]:
                weights[slot] = weight
                moments[slot] = weight * np.outer(vector, vector)
        cap = max(1., weights.sum())
        moments /= cap
        weights /= cap
        self._remove(block)
        if weights.any():
            self.contributions[block] = moments, weights
            self.moments += moments
            self.weights += weights
            self.support += weights > 0

    def read(self, axes, rotation):
        """匹配独立子图支持的低离散方向。每个查询和当前法向最多使用一次。"""
        pairs = []
        for slot in np.flatnonzero(self.support >= 2):
            values, vectors = np.linalg.eigh(self.moments[slot] / self.weights[slot])
            scatter = max(0., 1 - values[-1])
            if scatter > MAX_SCATTER:
                continue
            prior_axis = rotation.T @ vectors[:, -1]
            for index, axis in enumerate(axes):
                dot = float(prior_axis @ axis.normal)
                if abs(dot) >= MATCH_COS:
                    match = DirectionMatch(int(slot), prior_axis * np.sign(dot),
                                           axis.normal, int(self.support[slot]), float(scatter))
                    pairs.append((abs(dot), index, match))
        used_axes, used_slots = set(), set()
        result = []
        for _, index, match in sorted(pairs, key=lambda pair: pair[0], reverse=True):
            if index not in used_axes and match.slot not in used_slots:
                result.append(match)
                used_axes.add(index)
                used_slots.add(match.slot)
        return result

    def after_publish(self, frame, axes, rotation):
        """收集已发布帧的独立对齐观测，在块结束后向下一帧开放。"""
        block = frame // BLOCK_FRAMES
        self.committed = None
        if frame % SAMPLE_INTERVAL == 0 and axes and rotation is not None:
            if self.block_basis is None:
                self.block_basis = rotation.copy()
            transform = self.block_basis.T @ rotation
            self.samples.extend((frame, Axis(transform @ axis.normal, axis.coverage, axis.confidence))
                                for axis in axes)
        if (frame + 1) % BLOCK_FRAMES != 0:
            return
        if self.horizon is not None:
            self.oldest = max(0, block + 1 - self.horizon)
            for previous in tuple(self.contributions):
                if previous < self.oldest:
                    self._remove(previous)
        if self.samples:
            groups, directions = zip(*self.samples)
            encoded = cluster_axes(directions, groups, 2, BLOCK_FRAMES // SAMPLE_INTERVAL)
            if encoded:
                self.commit(block, encoded, self.block_basis)
                if block in self.contributions:
                    self.committed = block
        self.samples.clear()
        self.block_basis = None
