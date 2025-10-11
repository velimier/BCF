from __future__ import annotations
from typing import List, Tuple
import math, random

from .models import SecondaryBlock

def estimate_block_width_length(V: float, Omega: float) -> Tuple[float, float]:
    s = V ** (1.0 / 3.0)
    area = s * s
    r = 1.0 + max(0.0, (Omega - 1.0) / 0.67)
    length = math.sqrt(max(1e-6, r * area))
    width = max(1e-6, area / length)
    return width, length

def orepass_hangups(blocks: List[SecondaryBlock], bell_width: float, seed: int = 1234) -> dict:
    r = random.Random(seed)
    widths, masses = [], []
    for b in blocks:
        w, l = estimate_block_width_length(b.V, b.Omega)
        if r.random() < 0.5:
            w, l = l, w
        widths.append(w); masses.append(b.V)

    i = 0; n = len(widths); high = low = 0; hangup_tons = 0.0
    while i < n:
        span = 0.0; arch_idxs = []
        while i < n and span <= bell_width:
            span += widths[i]; arch_idxs.append(i); i += 1
        if span <= bell_width: break
        last = arch_idxs.pop(); span -= widths[last]
        if span >= 0.8 * bell_width:
            cnt = len(arch_idxs)
            if cnt < 3: p = 1.0
            elif cnt == 3: p = 0.95
            elif cnt == 4: p = 0.50
            elif cnt == 5: p = 0.05
            else: p = 0.0
            if r.random() < p:
                hangup_tons += sum(masses[j] for j in arch_idxs)
                if cnt <= 3: high += 1
                else: low += 1
    return {"n_high": high, "n_low": low, "total_hangup_tons": hangup_tons}

def kear_hangups(blocks: List[SecondaryBlock], bell_area: float, seed: int = 1234) -> dict:
    r = random.Random(seed); high = low = 0; hangup_tons = 0.0; batch = []
    for b in blocks:
        w, l = estimate_block_width_length(b.V, b.Omega)
        if r.random() < 0.5: w, l = l, w
        area = w * l; batch.append((area, b.V))
        if len(batch) == 25:
            areasum = sum(a for a, _ in batch)
            if areasum >= 0.4 * bell_area:
                hangup_tons += sum(v for _, v in batch); high += 1
            else:
                low += 1
            batch.clear()
    return {"n_high": high, "n_low": low, "total_hangup_tons": hangup_tons}
