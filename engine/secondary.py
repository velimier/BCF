from __future__ import annotations
from typing import List, Tuple
import math, random

from .models import RockMass, SecondaryRun, Defaults, PrimaryBlock, SecondaryBlock
from .strength import compute_IBS, IRS_to_IRSR, compute_RMS, block_strength

G_MPAPER_M = 9.80665 / 1e6

def caved_height(draw_height: float, swell_factor: float) -> float:
    F = max(1.0, float(swell_factor))
    return draw_height * F

def cave_pressure_MPa(density: float, Hc: float, width: float, swell_factor: float) -> float:
    if Hc <= 0.0:
        return 0.0
    ratio = max(1e-6, width / Hc)
    pairs = [(1.0, 0.44), (0.5, 0.30), (1/3, 0.23), (0.25, 0.20), (0.2, 0.18)]
    pairs = sorted(pairs, key=lambda x: x[0])
    if ratio >= pairs[0][0]:
        frac = pairs[0][1]
    elif ratio <= pairs[-1][0]:
        frac = pairs[-1][1]
    else:
        frac = pairs[-1][1]
        for i in range(len(pairs) - 1):
            r0, f0 = pairs[i + 1]
            r1, f1 = pairs[i]
            if r0 <= ratio <= r1:
                t = (ratio - r0) / (r1 - r0)
                frac = f0 + t * (f1 - f0)
                break
    F = max(1.0, float(swell_factor))
    rho_broken = density / F
    return frac * rho_broken * G_MPAPER_M * Hc

def draw_rate_factor(d_cm_day: float) -> float:
    return 0.66 * math.exp(0.023 * d_cm_day)

def pressure_factor(P_MPa: float) -> float:
    if P_MPa <= 1.0:
        return 1.0
    if P_MPa >= 12.0:
        return 0.5
    t = (P_MPa - 1.0) / (12.0 - 1.0)
    return 1.0 - 0.5 * t

def split_prob_from_Omega(Omega: float, with_joints: bool) -> float:
    table6 = [(0.999, 0.10), (2, 0.20), (3, 0.30), (4, 0.40), (5, 0.50),
              (6, 0.60), (7, 0.70), (8, 0.80), (9, 0.90), (10.0001, 1.0)]
    table7 = [(1, 20), (2, 40), (3, 60), (4, 80), (5, 100),
              (6, 100), (7, 100), (8, 100), (9, 100), (10.0001, 100)]
    t = table7 if with_joints else table6
    for thr, p in t:
        if Omega <= thr:
            return (p / 100.0) if p > 1.0 else p
    return (t[-1][1] / 100.0)

def cushioning_factor(fines_pct: float) -> float:
    keys = [5,10,15,20,25,30,35,40,50,60]
    vals = [0.95,0.90,0.85,0.80,0.75,0.70,0.60,0.50,0.40,0.30]
    if fines_pct <= 0:
        return 1.0
    for k, v in zip(keys, vals):
        if fines_pct <= k:
            return v
    return vals[-1]

def rounding_fines_pct(mu_deg: float) -> float:
    return (mu_deg / 5.0) + 3.0

def average_scatter_deg_from_jointsets(joints: list) -> float:
    if not joints:
        return 15.0
    vals = []
    for js in joints:
        vals.append(0.5 * (float(getattr(js, "dip_range", 10.0)) + float(getattr(js, "dip_dir_range", 10.0))))
    return sum(vals) / len(vals)

def run_secondary(prim_blocks: List[PrimaryBlock], rock: RockMass, sec: SecondaryRun, defaults: Defaults, mu_scatter_deg: float, primary_fines_ratio: float = 0.0):
    r = random.Random(defaults.seed or 1234)
    IBS = rock.IBS if rock.IBS is not None else compute_IBS(rock.IRS, rock.frac_freq, rock.frac_condition)
    IRSR = IRS_to_IRSR(rock.IRS)
    RMS = compute_RMS(rock.MRMR, rock.IRS, IRSR)

    Hc = caved_height(sec.draw_height, sec.swell_factor)
    P = cave_pressure_MPa(rock.density, Hc, sec.active_draw_width, sec.swell_factor)
    Fp = pressure_factor(P); Fr = draw_rate_factor(sec.rate_cm_day); Fc = 1.0

    cushioning_fines_pct = 100.0 * primary_fines_ratio + sec.added_fines_pct

    out: List[SecondaryBlock] = []
    sec_fines_mass = 0.0

    for blk in prim_blocks:
        stack = [(blk.V, blk.Omega, blk.joints_inside, sec.draw_height)]
        while stack:
            V, Omega, J, z = stack.pop()
            contains_joints = (J > 0)
            sigma_c = block_strength(V, contains_joints, rock.IRS, IBS, RMS)
            H_cycle = max(1.0, Fc * Fp * Fr * sigma_c)
            p = split_prob_from_Omega(Omega, with_joints=contains_joints)
            if V > 1.0: p *= cushioning_factor(cushioning_fines_pct)
            if r.random() < p:
                f = rounding_fines_pct(mu_scatter_deg) / 100.0
                sec_fines_mass += V * f
                childV = 0.5 * V * (1.0 - f)
                childOmega = max(1.0, 0.5 * Omega)
                new_z = z - H_cycle
                if new_z <= 0:
                    out.append(SecondaryBlock(V=childV, Omega=childOmega, joints_inside=J))
                    out.append(SecondaryBlock(V=childV, Omega=childOmega, joints_inside=J))
                else:
                    stack.append((childV, childOmega, J, new_z))
                    stack.append((childV, childOmega, J, new_z))
            else:
                new_z = z - H_cycle
                if new_z <= 0:
                    out.append(SecondaryBlock(V=V, Omega=Omega, joints_inside=J))
                else:
                    stack.append((V, Omega, J, new_z))

    r.shuffle(out)
    candidates = [i for i, b in enumerate(out) if b.V > 2.0]
    n_split = int(defaults.arching_pct * len(candidates))
    for idx in candidates[:n_split]:
        b = out[idx]
        childV = 0.5 * b.V; childO = max(1.0, 0.5 * b.Omega)
        out[idx] = SecondaryBlock(V=childV, Omega=childO, joints_inside=b.joints_inside)
        out.append(SecondaryBlock(V=childV, Omega=childO, joints_inside=b.joints_inside))

    total_mass = sum(b.V for b in out) + 1e-9
    secondary_fines_ratio = sec_fines_mass / total_mass
    return out, secondary_fines_ratio
