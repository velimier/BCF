from __future__ import annotations
from typing import Tuple
import math

def compute_IBS(IRS: float, frac_freq: float, frac_condition: int) -> float:
    ff_norm = max(0.0, min(1.0, frac_freq / 6.0))
    cond_norm = max(0.0, min(1.0, frac_condition / 40.0))
    severity = ff_norm * (1.0 - 0.5 * cond_norm)
    factor = 0.8 - (0.8 - 0.32) * severity
    return max(0.05 * IRS, factor * IRS)

def IRS_to_IRSR(IRS: float) -> int:
    bins = [
        (185, 20), (165, 18), (145, 16), (125, 14), (104, 12),
        (85, 10), (65, 8), (45, 6), (25, 4), (5, 2), (0, 0),
    ]
    for thr, rating in bins:
        if IRS > thr:
            return rating
    return 0

def compute_RMS(MRMR: float, IRS: float, IRSR: float) -> float:
    exponent = (MRMR - IRSR) / 80.0
    return 0.8 * IRS * (10.0 ** exponent)

def HB_mass_params(mi: float, RMR: float, IRS: float, RMS: float) -> Tuple[float, float]:
    m = mi * math.exp((RMR - 100.0) / 28.0)
    s = (RMS / max(1e-6, IRS)) ** 2
    return m, s

def block_strength(volume_m3: float, contains_joints: bool, IRS: float, IBS: float, RMS: float) -> float:
    v = max(1e-6, volume_m3)
    if not contains_joints:
        if v <= 1.0:
            return IRS - (IRS - IBS) * v / 1.0
        return IBS
    base = IRS - (IRS - IBS) * min(v, 1.0) / 1.0 if v <= 1.0 else IBS
    base *= 0.7
    if v >= 100.0:
        return RMS
    if base <= 0:
        return RMS
    k = math.log(max(1e-6, base / RMS)) / max(1e-6, (100.0 - 1.0))
    return max(RMS, base * math.exp(-k * (v - 1.0)))
