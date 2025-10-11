from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class RockMass:
    rock_type: str = "Unknown"
    MRMR: float = 65.0
    IRS: float = 120.0
    IBS: Optional[float] = None
    mi: float = 17.0
    frac_freq: float = 0.0
    frac_condition: int = 20
    density: float = 3200.0

@dataclass
class SpacingDist:
    type: str = "trunc_exp"
    min: float = 0.3
    mean: float = 1.0
    max_or_90pct: float = 3.0
    max_obs: Optional[float] = None

@dataclass
class JointSet:
    name: str
    mean_dip: float
    dip_range: float
    mean_dip_dir: float
    dip_dir_range: float
    spacing: SpacingDist
    JC: int = 20

@dataclass
class CaveFace:
    dip: float = 45.0
    dip_dir: float = 0.0
    stress_dip: float = 5.0
    stress_strike: float = 5.0
    stress_normal: float = 0.0
    allow_stress_fractures: bool = True
    spalling_pct: float = 0.0

@dataclass
class Defaults:
    LHD_cutoff_m3: float = 2.0
    seed: Optional[int] = 1234
    arching_pct: float = 0.12
    arch_stress_conc: float = 25.0
    stress_frac_trace_min: float = 10.0
    stress_frac_trace_mean: float = 20.0
    stress_frac_trace_max: float = 30.0
    tension_factor: float = 0.0

@dataclass
class SecondaryRun:
    draw_height: float = 150.0
    max_caving_height: float = 300.0
    swell_factor: float = 1.2
    active_draw_width: float = 45.0
    added_fines_pct: float = 0.0
    rate_cm_day: float = 20.0
    drawbell_upper_width: float = 8.0
    drawbell_lower_width: float = 6.0

@dataclass
class PrimaryBlock:
    V: float
    Omega: float
    joints_inside: int
    A: float
    lambda_max: float

@dataclass
class SecondaryBlock:
    V: float
    Omega: float
    joints_inside: int
