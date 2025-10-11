from __future__ import annotations
from typing import List
import math, random

from .models import RockMass, JointSet, CaveFace, Defaults, PrimaryBlock
from .distributions import sample_spacing
from .strength import compute_IBS, IRS_to_IRSR, compute_RMS

def _spacing_mean(sp):
    if isinstance(sp, dict):
        return float(sp.get("mean", 1.0))
    m = getattr(sp, "mean", None)
    return float(m) if m is not None else 1.0

def prob_from_JC(JC: int) -> float:
    xs = [0, 10, 20, 30, 40]
    ys = [0.05, 0.25, 0.50, 0.75, 0.92]
    JC = max(0, min(40, JC))
    for i in range(len(xs) - 1):
        if xs[i] <= JC <= xs[i + 1]:
            t = (JC - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]

def prob_weight_from_volume(V: float, JC: int) -> float:
    k = {15: 0.015, 20: 0.020, 25: 0.028, 30: 0.036}
    key = min(k.keys(), key=lambda j: abs(JC - j))
    return 1.0 - math.exp(-k[key] * max(0.0, V))

def shear_FOS(cave: CaveFace, JC: int) -> float:
    phi = math.radians(30.0)
    Cj = 0.015 * JC
    sigma_n = max(0.0, cave.stress_normal)
    tau = 0.5 * math.sqrt(cave.stress_dip ** 2 + cave.stress_strike ** 2)
    return (Cj + sigma_n * math.tan(phi)) / max(1e-6, tau)

def maybe_add_stress_fracture_set(rng: random.Random, rock: RockMass, cave: CaveFace) -> List[JointSet]:
    sigma1 = max(cave.stress_dip, cave.stress_strike, cave.stress_normal)
    IBS = rock.IBS if rock.IBS is not None else compute_IBS(rock.IRS, rock.frac_freq, rock.frac_condition)
    if IBS <= 0:
        return []
    F = sigma1 / IBS
    table = [
        (0.1, 0.01), (0.2, 0.03), (0.4, 0.05), (0.5, 0.07), (0.7, 0.10),
        (0.9, 0.20), (1.0, 0.30), (1.2, 0.50), (1.5, 1.00), (2.0, 10.00),
    ]
    if F >= 2.0 or F <= 0:
        return []
    Fvals = [f for f, _ in table]
    Svals = [s for _, s in table]
    if F < Fvals[0]:
        mean_spacing = Svals[0]
    else:
        mean_spacing = Svals[-1]
        for i in range(len(Fvals) - 1):
            if Fvals[i] <= F <= Fvals[i + 1]:
                t = (F - Fvals[i]) / (Fvals[i + 1] - Fvals[i])
                mean_spacing = Svals[i] + t * (Svals[i + 1] - Svals[i])
                break
    js = JointSet(
        name="StressFractures",
        mean_dip=cave.dip, dip_range=0.0,
        mean_dip_dir=cave.dip_dir, dip_dir_range=0.0,
        spacing=dict(type="trunc_exp", min=0.5 * mean_spacing, mean=mean_spacing, max_or_90pct=2.0 * mean_spacing),
        JC=0,
    )
    return [js]

def omega_from_dims(a: float, b: float, c: float) -> float:
    a, b, c = sorted([a, b, c], reverse=True)
    r = max(1.0, a / max(1e-6, c))
    return 1.0 + 0.67 * (r - 1.0)

def approximate_block_dims(rng: random.Random, sets: List[JointSet]) -> List[float]:
    dims = []
    for js in sets:
        dims.append(sample_spacing(rng, js.spacing))
    return [max(0.05, float(d)) for d in dims]

def generate_primary_blocks(n_blocks: int, rock: RockMass, joints: List[JointSet], cave: CaveFace, defaults: Defaults, seed: int = 1234) -> List[PrimaryBlock]:
    r = random.Random(seed)
    stress_sets = maybe_add_stress_fracture_set(r, rock, cave) if cave.allow_stress_fractures else []
    all_sets = list(joints) + stress_sets
    if len(all_sets) < 3:
        raise ValueError("At least 3 joint sets (including optional stress fractures) are required.")
    blocks: List[PrimaryBlock] = []
    for _ in range(n_blocks):
        weights = [1.0 / max(1e-6, _spacing_mean(js.spacing)) for js in all_sets]
        idxs = r.choices(range(len(all_sets)), weights=weights, k=3)
        idxs = list(dict.fromkeys(idxs))
        while len(idxs) < 3:
            idxs.append(r.randrange(0, len(all_sets)))
            idxs = list(dict.fromkeys(idxs))
        chosen = [all_sets[i] for i in idxs]
        a, b, c = sorted(approximate_block_dims(r, chosen), reverse=True)
        joints_inside = 0
        for js, dim in zip(chosen, [a, b, c]):
            P = prob_from_JC(js.JC)
            if shear_FOS(cave, js.JC) < 1.0:
                P = min(1.0, P + 0.20)
            V_tmp = a * b * c
            Pw = prob_weight_from_volume(V_tmp, max(15, min(30, js.JC)))
            if r.random() > max(P, Pw):
                ext = sample_spacing(r, js.spacing)
                if dim == a:
                    a += ext
                elif dim == b:
                    b += ext
                else:
                    c += ext
                joints_inside += 1
        V = max(1e-6, a * b * c)
        A = 2.0 * (a * b + b * c + c * a)
        lam = max(a, b, c)
        Omega = omega_from_dims(a, b, c)
        blocks.append(PrimaryBlock(V=V, Omega=Omega, joints_inside=joints_inside, A=A, lambda_max=lam))
    return blocks
