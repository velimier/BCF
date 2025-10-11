from __future__ import annotations
from typing import List
from .models import RockMass, CaveFace, PrimaryBlock, SecondaryBlock
from .strength import compute_IBS, IRS_to_IRSR, compute_RMS

def log_bins():
    bins = []; x = -2.0
    for _ in range(20):
        lo = 10 ** x; hi = 10 ** (x + 0.25)
        bins.append((lo, hi)); x += 0.25
    return bins

def distributions_from_blocks(blocks: List[PrimaryBlock] | List[SecondaryBlock]):
    bins = log_bins()
    freq_counts = [0] * 20; mass_counts = [0.0] * 20
    total_blocks = len(blocks); total_mass = sum(b.V for b in blocks) + 1e-9
    for b in blocks:
        V = b.V; idx = None
        for i, (lo, hi) in enumerate(bins):
            if lo <= V < hi:
                idx = i; break
        if idx is None: idx = 19 if V >= bins[-1][0] else 0
        freq_counts[idx] += 1; mass_counts[idx] += V
    cum_freq, cum_mass, linear_cum = [], [], []
    fc = 0; mc = 0.0
    for i in range(20):
        fc += freq_counts[i]; mc += mass_counts[i]
        cum_freq.append(100.0 * fc / max(1, total_blocks))
        cum_mass.append(100.0 * mc / total_mass)
    mc = 0.0
    for i in range(20):
        mc += mass_counts[i]; linear_cum.append(100.0 * mc / total_mass)
    return {
        "bins": bins, "freq_counts": freq_counts, "mass_counts": mass_counts,
        "cum_freq": cum_freq, "cum_mass": cum_mass, "linear_cum_mass": linear_cum,
        "max_volume": max((b.V for b in blocks), default=0.0),
        "avg_volume": (total_mass / max(1, total_blocks)) if blocks else 0.0,
        "avg_omega": (sum(b.Omega for b in blocks) / max(1, total_blocks)) if blocks else 0.0,
    }

def write_prm(path: str, rock: RockMass, cave: CaveFace, prim_blocks: List[PrimaryBlock], primary_fines_ratio: float):
    stats = distributions_from_blocks(prim_blocks)
    IBS = rock.IBS if rock.IBS is not None else compute_IBS(rock.IRS, rock.frac_freq, rock.frac_condition)
    IRSR = IRS_to_IRSR(rock.IRS); RMS = compute_RMS(rock.MRMR, rock.IRS, IRSR)
    with open(path, "w") as f:
        for b in prim_blocks:
            f.write(f"{b.V:.6f} {b.Omega:.6f} {b.joints_inside} {b.A:.6f} {b.lambda_max:.6f}\n")
        f.write("-1.0 -1.0 0\n")
        mass_pct_lt2 = 100.0 * sum(b.V for b in prim_blocks if b.V < 2.0) / (sum(b.V for b in prim_blocks) + 1e-9)
        f.write(f"{rock.rock_type:20s}{len(prim_blocks)} 0 {rock.IRS:.3f} {IBS:.3f} {RMS:.3f} {mass_pct_lt2:.3f} {primary_fines_ratio:.6f}\n")
        f.write(f"{cave.dip:.3f} {cave.dip_dir:.3f} {cave.stress_dip:.3f} {cave.stress_strike:.3f} {cave.stress_normal:.3f}\n")
        f.write(f"{stats['max_volume']:.6f} {stats['avg_volume']:.6f} {stats['avg_omega']:.6f} 0.0\n")
        for i in range(20):
            f.write(f"{stats['freq_counts'][i]}\n")
            f.write(f"{stats['mass_counts'][i]:.6f}\n")
            f.write(f"{stats['cum_freq'][i]:.3f}\n")
            f.write(f"{stats['cum_mass'][i]:.3f}\n")
            f.write(f"{stats['linear_cum_mass'][i]:.3f}\n")
            f.write(f"{stats['cum_freq'][i]:.3f}\n")
            f.write(f"{stats['cum_mass'][i]:.3f}\n")

def write_sec(path: str, rock: RockMass, cave: CaveFace, sec_blocks: List[SecondaryBlock], primary_fines_ratio: float, ratio_from_first_file: float = 1.0):
    class Pwrap:
        def __init__(self, V, Omega, joints_inside):
            self.V, self.Omega, self.joints_inside = V, Omega, joints_inside
    prim_wrapped = [Pwrap(b.V, b.Omega, b.joints_inside) for b in sec_blocks]
    stats = distributions_from_blocks(prim_wrapped)
    IBS = rock.IBS if rock.IBS is not None else compute_IBS(rock.IRS, rock.frac_freq, rock.frac_condition)
    IRSR = IRS_to_IRSR(rock.IRS); RMS = compute_RMS(rock.MRMR, rock.IRS, IRSR)
    with open(path, "w") as f:
        for b in sec_blocks:
            f.write(f"{b.V:.6f} {b.Omega:.6f} {b.joints_inside}\n")
        f.write("-1.0 -1.0 0\n")
        mass_pct_lt2 = 100.0 * sum(b.V for b in sec_blocks if b.V < 2.0) / (sum(b.V for b in sec_blocks) + 1e-9)
        f.write(f"{rock.rock_type:20s}{len(sec_blocks)} 1 {rock.IRS:.3f} {IBS:.3f} {RMS:.3f} {mass_pct_lt2:.3f} {primary_fines_ratio:.6f}\n")
        f.write(f"{cave.dip:.3f} {cave.dip_dir:.3f} {cave.stress_dip:.3f} {cave.stress_strike:.3f} {cave.stress_normal:.3f}\n")
        f.write(f"{stats['max_volume']:.6f} {stats['avg_volume']:.6f} {stats['avg_omega']:.6f} {ratio_from_first_file:.3f}\n")
        for i in range(20):
            f.write(f"{stats['freq_counts'][i]}\n")
            f.write(f"{stats['mass_counts'][i]:.6f}\n")
            f.write(f"{stats['cum_freq'][i]:.3f}\n")
            f.write(f"{stats['cum_mass'][i]:.3f}\n")
            f.write(f"{stats['linear_cum_mass'][i]:.3f}\n")
            f.write(f"{stats['cum_freq'][i]:.3f}\n")
            f.write(f"{stats['cum_mass'][i]:.3f}\n")
