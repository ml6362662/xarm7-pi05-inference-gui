"""Choose continuous revolute-joint coordinates; never reshape a target pose."""
import math
import numpy as np


def continuous_equivalent_angles(angles, reference, lower, upper):
    """Select q + 2*pi*k nearest the previous command inside existing bounds.

    Each joint's physical orientation is unchanged. This is not rate limiting,
    smoothing, interpolation, or clipping. No valid equivalent means an error.
    The caller resets reference from measured joints when entering servo mode.
    """
    q, ref, lo, hi = [np.asarray(a, dtype=float) for a in (angles, reference, lower, upper)]
    if any(a.shape != (7,) or not np.isfinite(a).all() for a in (q, ref, lo, hi)):
        raise ValueError("关节角、参考角与范围必须为7个有限值")
    if np.any(lo > hi):
        raise ValueError("关节范围无效")
    result = q.copy()
    period = 2 * math.pi
    for j in range(7):
        first = math.ceil((lo[j] - q[j]) / period)
        last = math.floor((hi[j] - q[j]) / period)
        if first > last:
            raise RuntimeError(f"J{j+1}逆解没有位于有效关节范围内的等价角度")
        k = min(range(first, last + 1),
                key=lambda n: (abs(q[j] + period*n - ref[j]), abs(n)))
        result[j] = q[j] + period*k
    return result
