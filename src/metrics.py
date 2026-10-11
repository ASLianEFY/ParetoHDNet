from __future__ import annotations
import numpy as np

def _points_2d(points, name: str, *, allow_empty: bool=True) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f'{name} must have shape (n, 2), got {pts.shape}')
    if not allow_empty and len(pts) == 0:
        raise ValueError(f'{name} must not be empty')
    if not np.all(np.isfinite(pts)):
        raise ValueError(f'{name} must contain only finite values')
    return pts

def _reference_point(ref) -> np.ndarray:
    ref = np.asarray(ref, dtype=float)
    if ref.shape != (2,):
        raise ValueError(f'ref must have shape (2,), got {ref.shape}')
    if not np.all(np.isfinite(ref)):
        raise ValueError('ref must contain only finite values')
    return ref

def pareto_filter(points: np.ndarray) -> np.ndarray:
    pts = _points_2d(points, 'points')
    if len(pts) == 0:
        return pts.copy()
    pts = np.unique(pts, axis=0)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        dominated = np.all(pts[i] <= pts, axis=1) & np.any(pts[i] < pts, axis=1)
        keep[dominated] = False
    return pts[keep]

def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    pts = pareto_filter(points)
    ref = _reference_point(ref)
    if len(pts) == 0:
        return 0.0
    pts = pts[(pts[:, 0] <= ref[0]) & (pts[:, 1] <= ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    cur_f2 = ref[1]
    for f1, f2 in pts:
        if f2 < cur_f2:
            hv += (ref[0] - f1) * (cur_f2 - f2)
            cur_f2 = f2
    return float(hv)

def igd(approx: np.ndarray, ref_front: np.ndarray) -> float:
    approx = _points_2d(approx, 'approx')
    ref_front = _points_2d(ref_front, 'ref_front', allow_empty=False)
    if len(approx) == 0:
        return float('inf')
    distances = np.linalg.norm(ref_front[:, None, :] - approx[None, :, :], axis=2)
    return float(np.min(distances, axis=1).mean())

def igd_plus(approx: np.ndarray, ref_front: np.ndarray) -> float:
    approx = _points_2d(approx, 'approx')
    ref_front = _points_2d(ref_front, 'ref_front', allow_empty=False)
    if len(approx) == 0:
        return float('inf')
    deltas = np.maximum(approx[None, :, :] - ref_front[:, None, :], 0.0)
    distances = np.linalg.norm(deltas, axis=2)
    return float(np.min(distances, axis=1).mean())

def spacing(front: np.ndarray) -> float:
    f = pareto_filter(front)
    if len(f) < 2:
        return 0.0
    f = f[np.argsort(f[:, 0])]
    distances = np.linalg.norm(f[1:] - f[:-1], axis=1)
    return float(np.sqrt(((distances - distances.mean()) ** 2).mean()))
