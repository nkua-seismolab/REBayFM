"""Kagan angle between two double-couple mechanisms.

Adapted from https://github.com/usgs/strec
(Kagan, Y., "Simplified algorithms for calculating double-couple rotation",
Geophysical Journal International, 171(1), 411-418).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from rebayfm.core.radiation import momtens


def get_kagan_angle(strike1, dip1, rake1, strike2, dip2, rake2) -> float:
    """Kagan angle (degrees) between two mechanisms given as strike/dip/rake."""
    tensor1 = momtens(1, strike1, dip1, rake1)
    tensor2 = momtens(1, strike2, dip2, rake2)
    return calc_theta(tensor1, tensor2)


def calc_theta(vm1: np.ndarray, vm2: np.ndarray) -> float:
    """Kagan angle (degrees) between two moment tensor matrices."""
    v1 = calc_eigenvec(vm1)
    v2 = calc_eigenvec(vm2)

    th = ang_from_r1r2(v1, v2)

    for j in range(3):
        k = (j + 1) % 3
        v3 = deepcopy(v2)
        v3[:, j] = -v3[:, j]
        v3[:, k] = -v3[:, k]
        x = ang_from_r1r2(v1, v3)
        if x < th:
            th = x
    return th * 180.0 / np.pi


def calc_eigenvec(tm: np.ndarray) -> np.ndarray:
    """Sorted eigenvector matrix of a moment tensor, right-handed."""
    vals, vecs = np.linalg.eigh(tm)
    inds = np.argsort(vals)
    vecs = vecs[:, inds]
    vecs[:, 2] = np.cross(vecs[:, 0], vecs[:, 1])
    return vecs


def ang_from_r1r2(r1: np.ndarray, r2: np.ndarray) -> float:
    """Rotation angle (radians) between two eigenvector matrices."""
    return np.arccos(np.clip((np.trace(np.dot(r1, r2.transpose())) - 1.0) / 2.0, -1, 1))
