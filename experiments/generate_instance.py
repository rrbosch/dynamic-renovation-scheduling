"""Generate a problem instance file with per-asset degradation parameters and env config.

Usage:
    # Interactive mode (no arguments — prompted for each parameter):
    python experiments/generate_instance.py

    # CLI mode:
    python experiments/generate_instance.py --n-assets 20 --network sioux_falls \
        --seed 0 --output instances/instance_001.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
import uuid

import numpy as np


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _lognormal_samples(rng: np.random.Generator, mean: float, sigma: float, n: int) -> np.ndarray:
    """Sample from LogNormal such that E[X] ≈ mean, with log-scale std sigma."""
    mu_log = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mean=mu_log, sigma=sigma, size=n)


# Base renovation duration (weeks) at zero length: e_ren_weeks = REN_BASE_WEEKS + L/5.
REN_BASE_WEEKS = 10.0


def lengths_mean_from_ongoing(
    avg_ongoing_projects: float, n_assets: int, e_fail_mean: float,
) -> float:
    """Mean asset length (m) so the portfolio sustains, in steady state, an average of
    ``avg_ongoing_projects`` assets under renovation at any time.

    Each asset cycles: degrade pristine→failure (mean ``e_fail_mean`` yr) then renovate
    (mean ``E[ren]`` yr). The fraction of time it is "ongoing" (under renovation) is
    ``E[ren] / (E[fail] + E[ren])``, so summed over the portfolio:

        P = N · E[ren] / (E[fail] + E[ren])   =>   E[ren] = P · E[fail] / (N − P)

    Inverting the length→duration map ``e_ren_weeks = REN_BASE_WEEKS + L/5`` then gives the
    required mean length. Mean-based (matches the file's in-expectation calibration); the
    realized count over the sampled, heterogeneous instance is reported by
    ``print_instance_stats``.

    Raises:
        ValueError: if ``P`` is outside the feasible range ``(P_min, n_assets)``, where
            ``P_min`` corresponds to the zero-length (base ``REN_BASE_WEEKS``) renovation.
    """
    P = float(avg_ongoing_projects)
    N = int(n_assets)
    if P <= 0.0:
        raise ValueError(f"avg_ongoing_projects must be > 0 (got {P}).")
    if P >= N:
        raise ValueError(
            f"avg_ongoing_projects={P} must be < n_assets={N} "
            "(cannot have >= N assets renovating simultaneously)."
        )

    e_ren_years = P * e_fail_mean / (N - P)
    e_ren_weeks = e_ren_years * 52.0
    if e_ren_weeks <= REN_BASE_WEEKS:
        e_ren_floor_years = REN_BASE_WEEKS / 52.0
        p_min = N * e_ren_floor_years / (e_fail_mean + e_ren_floor_years)
        raise ValueError(
            f"avg_ongoing_projects={P} implies a renovation of {e_ren_weeks:.2f} weeks, "
            f"below the {REN_BASE_WEEKS:.0f}-week base (would need negative length). "
            f"With n_assets={N} and e_fail_mean={e_fail_mean} yr the feasible range is "
            f"({p_min:.3f}, {N})."
        )

    return 5.0 * (e_ren_weeks - REN_BASE_WEEKS)


def generate_instance(
    n_assets: int,
    network: str,
    seed: int,
    lengths_mean_m: float = 200.0,
    lengths_cv: float = 0.5,
    alpha0_mean: float = 0.05,
    alpha0_sigma: float = 0.3,
    e_fail_cv: float = 0.10,
    e_fail_mean: float = 60.0,
    ren_noise_cv: float = 0.20,
    restrict_degrad_multiplier: float = 0.5,
    avg_ongoing_projects: float | None = None,
) -> dict:
    """
    Calibration targets:
      avg_ongoing_projects — if set, overrides lengths_mean_m: the mean asset length is
                          back-solved (via lengths_mean_from_ongoing) so the portfolio
                          sustains this many assets under renovation on average in steady
                          state. lengths_cv still controls spread around the derived mean.
      E[time to fail]   = e_fail_mean years (default 60), CV = e_fail_cv
                          Achieved by sampling e_fail ~ LogNormal(mean=e_fail_mean, sigma_log≈e_fail_cv)
                          then setting beta = alpha0 * e_fail so that beta/alpha0 = e_fail exactly.
      E[ren duration]   = 10 + length_i/5 weeks (base 10 weeks + 1 week per 5m of length)
                          mu_h_i = 1 / e_ren_i (years⁻¹).
      sigma_h           = ren_noise_cv * mu_h  (fixed noise-to-drift ratio across assets).
      asset_lengths_m   ~ LogNormal(mean=lengths_mean_m, CV=lengths_cv)
      alpha0            ~ LogNormal(mean=alpha0_mean, sigma_log=alpha0_sigma)

    Stochasticity levers:
      alpha0_mean   — higher value → tighter Gamma increments per step (same E[fail])
      alpha0_sigma  — 0 → all assets have identical alpha0 (homogeneous degradation shape)
      e_fail_cv     — 0 → all assets have identical expected lifetime
      ren_noise_cv  — 0 → renovation duration is deterministic (σ_h = 0)
    """
    rng = np.random.default_rng(seed)

    # --- Resolve mean length from target ongoing-projects count (overrides lengths_mean_m) ---
    if avg_ongoing_projects is not None:
        lengths_mean_m = lengths_mean_from_ongoing(
            avg_ongoing_projects, n_assets, e_fail_mean)

    # --- Asset lengths (metres) ---
    if lengths_cv == 0.0:
        lengths = np.full(n_assets, lengths_mean_m)
    else:
        sigma_log = np.sqrt(np.log(1 + lengths_cv ** 2))
        lengths = rng.lognormal(
            mean=np.log(lengths_mean_m) - 0.5 * sigma_log ** 2,
            sigma=sigma_log,
            size=n_assets,
        )

    # --- Degradation parameters ---
    # alpha0: controls within-asset variance of degradation increments (not mean rate)
    if alpha0_sigma == 0.0:
        alpha0 = np.full(n_assets, alpha0_mean)
    else:
        alpha0 = _lognormal_samples(rng, mean=alpha0_mean, sigma=alpha0_sigma, n=n_assets)

    # e_fail = beta/alpha0; LogNormal with mean=e_fail_mean years and CV=e_fail_cv
    if e_fail_cv == 0.0:
        e_fail = np.full(n_assets, e_fail_mean)
    else:
        e_fail_sigma_log = np.sqrt(np.log(1.0 + e_fail_cv ** 2))
        e_fail = _lognormal_samples(rng, mean=e_fail_mean, sigma=e_fail_sigma_log, n=n_assets)
    beta = alpha0 * e_fail

    # --- Renovation parameters ---
    # E[ren duration] = REN_BASE_WEEKS + length_i / 5 weeks
    e_ren_weeks = REN_BASE_WEEKS + lengths / 5.0
    e_ren_years = e_ren_weeks / 52.0
    mu_h    = 1.0 / e_ren_years
    sigma_h = ren_noise_cv * mu_h

    # --- Cost parameters derived from lengths ---
    c_ren = 50_000.0 * lengths  # €/asset
    c_rep = 25_000.0 * lengths  # €/asset

    d_init = rng.uniform(0.3, 0.8, size=n_assets)

    return {
        'schema_version': 1,
        'n_assets': n_assets,
        'network': network,
        'generation_seed': seed,
        'd_init':               d_init.tolist(),
        'alpha0':               alpha0.tolist(),
        'beta':                 beta.tolist(),
        'mu_h':                 mu_h.tolist(),
        'sigma_h':              sigma_h.tolist(),
        'asset_lengths_m':      lengths.tolist(),
        'c_ren':                c_ren.tolist(),
        'c_rep':                c_rep.tolist(),
        'restrict_degrad_multiplier': restrict_degrad_multiplier,
        'lengths_mean_m_resolved': float(lengths_mean_m),
    }


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def print_instance_stats(inst: dict, target_ongoing: float | None = None) -> None:
    """Print per-asset statistics table assuming baseline conditions.

    Baseline assumptions:
      - No load restriction (ell = 0), so alpha = alpha0
      - Renovation triggered at full degradation (d = d_fail = 1.0)
      - Post-renovation d resets to 0 (new cycle starts)

    Formulas:
      E[time to fail]      = beta / alpha0   (Gamma process, mean rate = alpha0/beta)
      E[renovation time]   = 1 / mu_h        (BM with drift -mu_h, first passage from h=1 to 0)
      Downtime %           = E[ren] / (E[fail] + E[ren]) * 100
    """
    alpha0  = np.array(inst['alpha0'])
    beta    = np.array(inst['beta'])
    mu_h    = np.array(inst['mu_h'])

    e_fail       = beta / alpha0           # years
    e_ren_years  = 1.0 / mu_h             # years
    e_ren_weeks  = e_ren_years * 52.0     # weeks (more readable)
    pct_down = 100.0 * e_ren_years / (e_fail + e_ren_years)

    col_w = (6, 20, 18, 12)
    header = (
        f"{'Asset':>{col_w[0]}}  "
        f"{'E[fail] (years)':>{col_w[1]}}  "
        f"{'E[ren] (weeks)':>{col_w[2]}}  "
        f"{'Downtime %':>{col_w[3]}}"
    )
    sep = '-' * len(header)
    print("\nPer-asset statistics (baseline: no load, no restriction):")
    print(sep)
    print(header)
    print(sep)
    for i in range(inst['n_assets']):
        print(
            f"{i:>{col_w[0]}}  "
            f"{e_fail[i]:>{col_w[1]}.1f}  "
            f"{e_ren_weeks[i]:>{col_w[2]}.1f}  "
            f"{pct_down[i]:>{col_w[3]}.2f}"
        )
    print(sep)
    print(
        f"{'mean':>{col_w[0]}}  "
        f"{e_fail.mean():>{col_w[1]}.1f}  "
        f"{e_ren_weeks.mean():>{col_w[2]}.1f}  "
        f"{pct_down.mean():>{col_w[3]}.2f}"
    )
    print(sep)

    # Implied average number of ongoing renovation projects = Σ (ongoing fraction).
    implied_ongoing = float((pct_down / 100.0).sum())
    target_note = (f"  (target was {target_ongoing:.3f})"
                   if target_ongoing is not None else "")
    print(f"Implied avg. ongoing projects = {implied_ongoing:.3f}{target_note}")
    print(sep)


# ---------------------------------------------------------------------------
# Interactive prompting
# ---------------------------------------------------------------------------

def _prompt_value(label: str, description: str, default, type_fn,
                  lo=None, lo_open: bool = False,
                  hi=None, hi_open: bool = False,
                  choices=None):
    """Prompt the user for a single parameter value, with validation.

    Args:
        label:       Short parameter name shown to the user.
        description: One-line explanation of what the parameter does.
        default:     Default value (shown and used on empty input). None = required.
        type_fn:     Callable to convert the raw string (e.g. int, float, str).
        lo:          Lower bound (inclusive unless lo_open=True).
        lo_open:     If True, lower bound is exclusive (strict >).
        hi:          Upper bound (inclusive unless hi_open=True).
        hi_open:     If True, upper bound is exclusive (strict <).
        choices:     Exhaustive list of allowed values (overrides bounds display).
    """
    # Build bounds / choices annotation
    if choices is not None:
        constraint = f"choices: {choices}"
    else:
        parts = []
        if lo is not None:
            bracket = '(' if lo_open else '['
            parts.append(f"{bracket}{lo}")
        if hi is not None:
            bracket = ')' if hi_open else ']'
            parts.append(f"{hi}{bracket}")
        constraint = ', '.join(parts) if parts else ''

    default_hint = "(required)" if default is None else f"default: {default}"

    while True:
        print(f"\n  {label}")
        print(f"    {description}")
        if constraint:
            print(f"    bounds: {constraint}")
        print(f"    {default_hint}")
        raw = input("    > ").strip()

        if raw == '':
            if default is None:
                print("    This parameter is required — please enter a value.")
                continue
            return default

        # Type conversion
        try:
            val = type_fn(raw)
        except (ValueError, TypeError):
            print(f"    Expected {type_fn.__name__} — try again.")
            continue

        # Choices validation
        if choices is not None and val not in choices:
            print(f"    Must be one of {choices} — try again.")
            continue

        # Bounds validation
        if lo is not None:
            if (lo_open and val <= lo) or (not lo_open and val < lo):
                op = '>' if lo_open else '>='
                print(f"    Value must be {op} {lo} — try again.")
                continue
        if hi is not None:
            if (hi_open and val >= hi) or (not hi_open and val > hi):
                op = '<' if hi_open else '<='
                print(f"    Value must be {op} {hi} — try again.")
                continue

        return val


def interactive_mode() -> types.SimpleNamespace:
    """Interactively prompt the user for all instance parameters.

    Returns a SimpleNamespace with the same fields used by _parse_args().
    """
    print("=" * 60)
    print("  Infrastructure Instance Generator — Interactive Mode")
    print("  Press Enter to accept the default value shown.")
    print("=" * 60)

    # ---- Output ----
    print("\n--- Output ---")
    output = _prompt_value(
        "output",
        "Path to write the instance JSON file (e.g. instances/my_instance.json).",
        default=None, type_fn=str,
    )

    # ---- Instance setup ----
    print("\n--- Instance setup ---")
    n_assets = _prompt_value(
        "n_assets",
        "Number of infrastructure assets (road edges) in the portfolio.",
        default=20, type_fn=int, lo=1,
    )
    network = _prompt_value(
        "network",
        "Traffic network used for travel-time cost computation.",
        default='sioux_falls', type_fn=str, choices=['sioux_falls'],
    )
    seed = _prompt_value(
        "seed",
        "Random seed — fix for reproducible instances.",
        default=0, type_fn=int,
    )

    # ---- MDP horizon & discounting ----
    print("\n--- MDP horizon & discounting ---")
    years = _prompt_value(
        "years",
        "Planning horizon in years. Default 60 yr.",
        default=60.0, type_fn=float, lo=0.0, lo_open=True,
    )
    dt = _prompt_value(
        "dt",
        "Epoch length in years (0.5 = half-year steps).",
        default=0.5, type_fn=float, lo=0.0, lo_open=True,
    )
    t_tail_years = _prompt_value(
        "t_tail_years",
        "Evaluation tail length in years (extra horizon simulated past T so terminal costs "
        "aren't truncated). 0 = auto (1x expected lifespan, i.e. = e_fail_mean).",
        default=0.0, type_fn=float, lo=0.0,
    )
    if t_tail_years <= 0.0:
        t_tail_years = None   # resolved to e_fail_mean in main()
    gamma = _prompt_value(
        "gamma",
        "Annual discount factor. Per-epoch value is derived as gamma ** dt.",
        default=0.97, type_fn=float, lo=0.0, lo_open=True, hi=1.0, hi_open=True,
    )

    # ---- Asset physics ----
    print("\n--- Asset physics ---")
    d_fail = _prompt_value(
        "d_fail",
        "Condition value at which an asset is considered failed (d >= d_fail). "
        "1.0 = fail only at maximum degradation.",
        default=1.0, type_fn=float, lo=0.0, lo_open=True, hi=1.0,
    )
    eta_ren = _prompt_value(
        "eta_ren",
        "Road capacity fraction during active renovation (0.05 = 5% of normal). "
        "Lower → heavier traffic penalty while renovating.",
        default=0.05, type_fn=float, lo=0.0, lo_open=True, hi=1.0, hi_open=True,
    )
    eta_load = _prompt_value(
        "eta_load",
        "Road capacity fraction under a load restriction (0.50 = half capacity). "
        "Reduces degradation rate but also imposes a traffic cost.",
        default=0.50, type_fn=float, lo=0.0, lo_open=True, hi=1.0, hi_open=True,
    )
    restrict_degrad_multiplier = _prompt_value(
        "restrict_degrad_multiplier",
        "Multiplier on degradation shape rate α under load restriction (analogous to eta_load for capacity). "
        "0.5 → rate halved; 1.0 → restriction has no effect on degradation; 0.0 → degradation fully stopped.",
        default=0.5, type_fn=float, lo=0.0, hi=1.0,
    )
    delta_repair = _prompt_value(
        "delta_repair",
        "Condition improvement (Δd) applied by one repair action. "
        "At most one repair per renovation cycle is allowed.",
        default=0.1, type_fn=float, lo=0.0, lo_open=True, hi=1.0,
    )

    # ---- Degradation stochasticity ----
    print("\n--- Degradation stochasticity ---")
    print("  Tip: set alpha0_sigma=0, e_fail_cv=0, ren_noise_cv=0 for a fully predictable instance.")
    alpha0_mean = _prompt_value(
        "alpha0_mean",
        "Mean Gamma shape rate α₀ across assets. Higher → tighter (less noisy) degradation "
        "increments per step while keeping E[time to fail] = 60 yr unchanged.",
        default=0.05, type_fn=float, lo=0.0, lo_open=True,
    )
    alpha0_sigma = _prompt_value(
        "alpha0_sigma",
        "Log-scale std of α₀ across assets. 0 → all assets have identical shape rate "
        "(homogeneous degradation variance); larger → more spread.",
        default=0.3, type_fn=float, lo=0.0,
    )
    e_fail_mean = _prompt_value(
        "e_fail_mean",
        "Expected lifespan of a new (pristine) asset in years. "
        "Sets E[time to failure] = e_fail_mean, achieved via beta = alpha0 * e_fail_mean.",
        default=60.0, type_fn=float, lo=0.0, lo_open=True,
    )
    e_fail_cv = _prompt_value(
        "e_fail_cv",
        "Coefficient of variation of time-to-fail distribution across assets. "
        "0 → all assets have identical expected lifetime (e_fail_mean yr).",
        default=0.10, type_fn=float, lo=0.0,
    )
    ren_noise_cv = _prompt_value(
        "ren_noise_cv",
        "Renovation duration noise as a fraction of the drift: σ_h = cv · μ_h. "
        "0 → renovation duration is fully deterministic.",
        default=0.20, type_fn=float, lo=0.0,
    )

    # ---- Asset lengths ----
    print("\n--- Asset lengths ---")
    avg_ongoing_projects = _prompt_value(
        "avg_ongoing_projects",
        "Target avg. number of assets under renovation in steady state. "
        "If > 0, the mean asset length is back-solved from n_assets and e_fail_mean "
        "(overrides lengths_mean_m). 0 = disabled (specify lengths_mean_m directly).",
        default=0.0, type_fn=float, lo=0.0,
    )
    if avg_ongoing_projects > 0.0:
        lengths_mean_m = None   # derived in generate_instance from the target
    else:
        avg_ongoing_projects = None
        lengths_mean_m = _prompt_value(
            "lengths_mean_m",
            "Mean asset length in metres. Drives renovation cost (50k €/m), repair cost (25k €/m), "
            "and expected renovation duration (10 wk + L/5 wk).",
            default=2000.0, type_fn=float, lo=0.0, lo_open=True,
        )
    lengths_cv = _prompt_value(
        "lengths_cv",
        "Coefficient of variation of asset lengths. 0 → all assets identical length.",
        default=0.5, type_fn=float, lo=0.0,
    )

    # ---- Cost & traffic ----
    print("\n--- Cost & traffic ---")
    vot = _prompt_value(
        "vot",
        "Value of time in €/vehicle-hour. Scales the travel-time component of cost.",
        default=10.76, type_fn=float, lo=0.0, lo_open=True,
    )
    traffic_cost_factor = _prompt_value(
        "traffic_cost_factor",
        "Multiplicative factor on raw Sioux Falls traffic costs. "
        "Use < 1 to down-scale if raw costs dominate other cost terms.",
        default=1.0, type_fn=float, lo=0.0,
    )
    risk_base = _prompt_value(
        "risk_base",
        "Base risk cost in €/m/year per epoch spent in failure. "
        "Total risk per epoch = risk_base · dt · Σ n_fail_i · L_i.",
        default=10_000.0, type_fn=float, lo=0.0,
    )

    print()
    return types.SimpleNamespace(
        output=output,
        n_assets=n_assets,
        network=network,
        seed=seed,
        years=years,
        dt=dt,
        t_tail_years=t_tail_years,
        gamma=gamma,
        d_fail=d_fail,
        eta_ren=eta_ren,
        eta_load=eta_load,
        restrict_degrad_multiplier=restrict_degrad_multiplier,
        delta_repair=delta_repair,
        alpha0_mean=alpha0_mean,
        alpha0_sigma=alpha0_sigma,
        e_fail_mean=e_fail_mean,
        e_fail_cv=e_fail_cv,
        ren_noise_cv=ren_noise_cv,
        lengths_mean_m=lengths_mean_m,
        lengths_cv=lengths_cv,
        avg_ongoing_projects=avg_ongoing_projects,
        vot=vot,
        traffic_cost_factor=traffic_cost_factor,
        risk_base=risk_base,
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> types.SimpleNamespace:
    parser = argparse.ArgumentParser(description='Generate a problem instance file.')
    parser.add_argument('--n-assets', type=int, default=20)
    parser.add_argument('--network', type=str, default='sioux_falls')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the instance JSON (required).')
    parser.add_argument('--years', type=float, default=60.0,
                        help='Planning horizon in years. T is derived as round(years/dt).')
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--t-tail-years', type=float, default=None,
                        help='Evaluation tail length in years (horizon simulated past T). '
                             'Default = e_fail_mean (1x expected lifespan).')
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--d-fail', type=float, default=1.0)
    parser.add_argument('--eta-ren', type=float, default=0.05)
    parser.add_argument('--eta-load', type=float, default=0.50)
    parser.add_argument('--restrict-degrad-multiplier', type=float, default=0.9,
                        help='Multiplier on degradation shape rate α under load restriction (0–1). '
                             '0.5 → rate halved; 1.0 → no effect; 0.0 → fully stopped.')
    parser.add_argument('--delta-repair', type=float, default=0.1)
    parser.add_argument('--vot', type=float, default=10.76,
                        help='Value of time (euros per vehicle-hour).')
    parser.add_argument('--lengths-mean-m', type=float, default=2000.0,
                        help='Mean asset length (m).')
    parser.add_argument('--lengths-cv', type=float, default=0.5,
                        help='CV for asset lengths. 0 = homogeneous.')
    parser.add_argument('--avg-ongoing-projects', type=float, default=None,
                        help='Target average number of assets under renovation in steady '
                             'state. If set, OVERRIDES --lengths-mean-m: the mean length is '
                             'back-solved from n_assets and e_fail_mean. lengths_cv still '
                             'controls spread.')
    parser.add_argument('--traffic-cost-factor', type=float, default=1.0,
                        help='Multiplicative factor on raw Sioux Falls traffic costs.')
    parser.add_argument('--risk-base', type=float, default=10_000.0,
                        help='Base risk rate (euros/m/year) per failure epoch.')
    parser.add_argument('--alpha0-mean', type=float, default=0.05,
                        help='Mean Gamma shape rate alpha0. Higher = less noisy degradation.')
    parser.add_argument('--alpha0-sigma', type=float, default=0.3,
                        help='Log-scale std of alpha0 across assets. 0 = homogeneous.')
    parser.add_argument('--e-fail-mean', type=float, default=60.0,
                        help='Expected lifespan of a new asset in years.')
    parser.add_argument('--e-fail-cv', type=float, default=0.10,
                        help='CV of time-to-fail distribution. 0 = identical lifetimes.')
    parser.add_argument('--ren-noise-cv', type=float, default=0.20,
                        help='Renovation duration noise as fraction of drift. 0 = deterministic.')
    a = parser.parse_args()
    return types.SimpleNamespace(
        output=a.output,
        n_assets=a.n_assets,
        network=a.network,
        seed=a.seed,
        years=a.years,
        dt=a.dt,
        t_tail_years=a.t_tail_years,
        gamma=a.gamma,
        d_fail=a.d_fail,
        eta_ren=a.eta_ren,
        eta_load=a.eta_load,
        restrict_degrad_multiplier=a.restrict_degrad_multiplier,
        delta_repair=a.delta_repair,
        alpha0_mean=a.alpha0_mean,
        alpha0_sigma=a.alpha0_sigma,
        e_fail_mean=a.e_fail_mean,
        e_fail_cv=a.e_fail_cv,
        ren_noise_cv=a.ren_noise_cv,
        lengths_mean_m=a.lengths_mean_m,
        lengths_cv=a.lengths_cv,
        avg_ongoing_projects=a.avg_ongoing_projects,
        vot=a.vot,
        traffic_cost_factor=a.traffic_cost_factor,
        risk_base=a.risk_base,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) == 1:
        args = interactive_mode()
    else:
        args = _parse_args()

    if args.output is None:
        raise SystemExit("error: --output is required in CLI mode (no default).")

    avg_ongoing_projects = getattr(args, 'avg_ongoing_projects', None)
    # lengths_mean_m may be None on the interactive "use ongoing-projects target" path;
    # fall back to the function default so the call is well-formed (it's overridden anyway).
    lengths_mean_m = args.lengths_mean_m if args.lengths_mean_m is not None else 2000.0

    inst = generate_instance(
        args.n_assets, args.network, args.seed,
        lengths_mean_m=lengths_mean_m,
        lengths_cv=args.lengths_cv,
        alpha0_mean=args.alpha0_mean,
        alpha0_sigma=args.alpha0_sigma,
        e_fail_mean=args.e_fail_mean,
        e_fail_cv=args.e_fail_cv,
        ren_noise_cv=args.ren_noise_cv,
        restrict_degrad_multiplier=args.restrict_degrad_multiplier,
        avg_ongoing_projects=avg_ongoing_projects,
    )

    # Resolved mean length actually used (back-solved when a target was given).
    resolved_lengths_mean_m = inst.pop('lengths_mean_m_resolved')
    if avg_ongoing_projects is not None:
        print(f"  avg_ongoing_projects={avg_ongoing_projects} -> derived "
              f"lengths_mean_m={resolved_lengths_mean_m:.1f} m "
              f"(overrides --lengths-mean-m)")

    # Evaluation tail length (years). Default = e_fail_mean (1x expected lifespan).
    t_tail_years = getattr(args, 't_tail_years', None)
    t_tail_auto = t_tail_years is None
    if t_tail_auto:
        t_tail_years = float(args.e_fail_mean)

    inst.update({
        'years': args.years,
        'dt': args.dt,
        'T_tail': float(t_tail_years),
        'gamma': args.gamma,
        'd_fail': args.d_fail,
        'eta_ren': args.eta_ren,
        'eta_load': args.eta_load,
        'restrict_degrad_multiplier': args.restrict_degrad_multiplier,
        'delta_repair': args.delta_repair,
        'vot': args.vot,
        'traffic_cost_factor': args.traffic_cost_factor,
        'risk_base': args.risk_base,
        'allow_repair': True,
        'allow_restrict': True,
        'instance_id': str(uuid.uuid4()),
        'schema_version': 5,
        'comments': {
            'lengths_mean_m': resolved_lengths_mean_m,
            'lengths_cv': args.lengths_cv,
            'avg_ongoing_projects': avg_ongoing_projects,
            'alpha0_mean': args.alpha0_mean,
            'alpha0_sigma': args.alpha0_sigma,
            'e_fail_mean': args.e_fail_mean,
            'e_fail_cv': args.e_fail_cv,
            'ren_noise_cv': args.ren_noise_cv,
            't_tail_auto': t_tail_auto,
        },
    })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(inst, f, indent=2)

    print(f"Generated instance -> {args.output}")
    print(f"  n_assets={inst['n_assets']}, network={inst['network']!r}, seed={inst['generation_seed']}")
    _tail_note = " (= 1x e_fail_mean)" if t_tail_auto else ""
    print(f"  T_tail={t_tail_years:.1f} yr{_tail_note}")
    print_instance_stats(inst, target_ongoing=avg_ongoing_projects)


if __name__ == '__main__':
    main()
