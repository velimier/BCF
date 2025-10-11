from __future__ import annotations
import math, random

def sample_truncated_exponential(rng: random.Random, mean: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    lam = max(1e-6, 1.0 / max(1e-6, (mean - 0.2 * lo)))
    for _ in range(10000):
        x = rng.expovariate(lam)
        v = lo + x
        if v <= hi:
            return v
    return lo + rng.random() * (hi - lo)

def sample_normal_range(rng: random.Random, lo: float, mean: float, hi: float) -> float:
    sigma = max(1e-6, (hi - lo) / 6.0)
    x = rng.gauss(mean, sigma)
    return max(lo, min(hi, x))

def sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return lo + rng.random() * (hi - lo)

def sample_lognormal_capped(rng: random.Random, p10: float, p90: float, max_obs: float) -> float:
    import math
    z10, z90 = -1.2815515655446004, 1.2815515655446004
    y10, y90 = math.log(max(1e-6, p10)), math.log(max(1e-6, p90))
    sigma = (y90 - y10) / (z90 - z10)
    mu = y10 - sigma * z10
    while True:
        v = rng.lognormvariate(mu, sigma)
        if v <= max_obs:
            return v

def sample_spacing(rng: random.Random, dist) -> float:
    if not isinstance(dist, dict):
        dist = getattr(dist, "__dict__", {})
    t = dist.get("type", "trunc_exp")
    lo = float(dist.get("min", 0.1))
    m = float(dist.get("mean", max(lo + 1e-6, lo * 1.2)))
    hi = float(dist.get("max_or_90pct", m * 3.0))
    if hi <= lo:
        hi = max(lo + 1e-3, lo * 1.05)
    if not (lo < m < hi):
        m = lo + 0.35 * (hi - lo)

    if t == "trunc_exp":
        return sample_truncated_exponential(rng, m, lo, hi)
    elif t == "normal":
        return sample_normal_range(rng, lo, m, hi)
    elif t == "uniform":
        return sample_uniform(rng, lo, hi)
    elif t == "lognormal":
        max_obs = float(dist.get("max_obs", hi * 3.0))
        return sample_lognormal_capped(rng, m * 0.6, m * 1.4, max_obs)
    else:
        return sample_truncated_exponential(rng, m, lo, hi)
