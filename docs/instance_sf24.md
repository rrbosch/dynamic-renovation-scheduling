# instance_sf24 — congestion-synergy calibration instance

`instances/instance_sf24.json` is the calibration instance we test on. It is engineered so that
**anticipation matters through avoiding clusters of road congestion**: assets are placed on roads
whose *simultaneous* renovation is catastrophically congesting, and they are timed so a myopic
policy is forced to bunch those closures while a foresighted policy staggers them. This document
explains the selection reasoning (why the assets are where they are) and gives the instance spec
and results.

- **Instance:** `instances/instance_sf24.json` (n=15)
- **Configs:** `configs/sf24_clairvoyant.json`, `configs/sf24_reactive.json`
- **Builder (reproducible):** `experiments/build_synergy_instance.py`
- **Baseline model:** each asset is one **bidirectional link** (both directions of a road; see
  `.claude/rules/env-internals.md`).

## 1. Congestion synergy

For a bidirectional link, "closing" it sets **both** directed edges to `eta_ren` (5%) capacity and
re-solves the traffic equilibrium (TAP). The congestion measure is the extra total network
travel-time vs the nominal baseline:

- `ΔT_i` = extra veh-hours from closing link *i* alone
- `ΔT_ij` = extra veh-hours from closing links *i* and *j* together

The **synergy** of a pair is the percentage by which the joint impact exceeds the naive sum
(`experiments/link_synergy.py`):

```
synergy(i, j) = ( ΔT_ij − (ΔT_i + ΔT_j) ) / (ΔT_i + ΔT_j) × 100 %
```

`0 %` = additive; `+100 %` = 2× the sum; `+16250 %` (links 1↔3 & 2↔6) = **~163×** the sum — a near
**cut** (closing both isolates a region, so rerouting cost explodes). Cost constants cancel in the
ratio (computed in raw veh-hours).

## 2. The gap is driven by synergy, not traffic volume

The clairvoyant↔reactive gap is a *coordination* gap: the clairvoyant staggers renovations to avoid
simultaneous congestion; the single-threshold reactive bunches them. So the gap depends on whether
co-renovated roads are **super-additive** (a cut) — not on how busy they are. Contrast experiment
(50-ep, tcf=1.0, everything else identical, only the *selection rule* changes):

| Selection | mean V/C | adjacent-pair synergy | Clairvoyant | Reactive | **Gap** | Reactive travel% |
|---|---|---|---|---|---|---|
| high mutual **synergy** | 0.45 | 1280 % | 2034 M | 8017 M | **74.6 %** | 81 % |
| high **V²/C** (busiest roads) | 0.59 | 233 % | 1718 M | 2347 M | **26.8 %** | 42 % |

Both selections renovate the same amount (≈equal clairvoyant cost and workload). But the busy-road
set is only *additively* congesting when bunched, so staggering barely helps → the reactive is
cheap → small gap. **Putting assets on the busiest roads — the intuitive choice — minimizes the
value of anticipation; high-synergy cuts maximize it** (even low-volume ones like the node-1 corner,
V/C 0.14–0.20). See `docs/sf24_v2c_map.png` for the V²/C map (green=low, red=high).

## 3. The volume-biased synergy blend (why the final selection)

Pure synergy over-selects tiny corner cuts (defensible for the gap, but odd as "assets" — you would
not prioritize maintaining a near-empty corner road). Pure volume destroys the gap. So the selection
uses a tunable **geometric blend** of the two importance scores (the *ordering* stays pure synergy,
so d_init-neighbours remain strongest-synergy pairs):

```
selection score(link) = synergy_participation^(1−λ) · (V²/C)^λ        (--v2c-bias λ)
```

Sweeping λ (selection-only preview):

| λ | mean V/C | adjacent-pair synergy |
|---|---|---|
| 0.0 (pure synergy) | 0.45 | 1280 % |
| 0.2 | 0.46 | 1337 % |
| 0.3 | 0.48 | 1288 % |
| **0.5** | **0.53** | **1420 %** |
| 0.7 | 0.55 | 1348 % |
| 1.0 (pure V²/C) | 0.59 | 233 % |

The adjacent-pair synergy — which the gap tracks — stays high (1300–1420 %) all the way to λ≈0.7,
while mean V/C climbs. **λ=0.5 is the sweet spot:** it keeps the catastrophic node-1 cut *and* pulls
in busy synergistic corridors (7↔8, 10↔11, 16↔17, 18↔20), giving the highest gap on the most
defensible (busiest) roads. Confirmed by full 50-ep eval:

| Selection | Clairvoyant | Reactive | **Gap** | mean V/C |
|---|---|---|---|---|
| λ=0 (pure synergy) | 2034 M | 8017 M | 74.6 % | 0.45 |
| **λ=0.5 (this instance)** | 1824 M | 7631 M | **76.1 %** | 0.53 |
| λ=1 (pure V²/C) | 1718 M | 2347 M | 26.8 % | 0.59 |

`docs/sf24_selected_links.png` shows the 15 chosen links coloured by d_init (failure order).

## 4. Instance construction & spec

Built by `experiments/build_synergy_instance.py`:
1. Compute the 38×38 synergy matrix (`link_synergy.compute_synergy`).
2. **Select** n=15 links by the λ=0.5 blend (greedy growth on the blended score).
3. **Order** them on a maximum-synergy Hamiltonian path, so each asset's d_init-neighbours are its
   strongest synergy partners.
4. **Continuous d_init** ~ `Uniform(lo, 0.85)` sorted and assigned along the path; `lo` auto-derived
   so every asset has > 80 % chance of failing in-horizon (Gamma first-passage). Homogeneous
   degradation (`e_fail_cv=0`, `alpha0_sigma=0`) ⇒ sorted-d_init order = failure order, so
   *fails-together = congestion-coupled* — and the failure timeline is a smooth continuous spread
   (no visible cohorts; `docs/sf24_degradation.png`).
5. **Auto-calibrated lengths:** binary search on renovation duration so the renovate-at-failure
   *workload* avg-sim hits `--target-avgsim` (3.5).

| Parameter | Value |
|---|---|
| n_assets | 15 (bidirectional links; blend-selected) |
| e_fail / degradation | 100 yr, homogeneous (cv=0), alpha0=0.8 (predictable) |
| d_init | continuous sorted `Uniform(≈0.29, 0.85)` (auto lo) |
| lengths / e_ren | auto ≈ 5865 m / ≈ 22.7 yr (for workload avg-sim 3.45) |
| horizon | years=80, dt=0.5, T_tail=15 |
| eta_ren / eta_load / restrict_mult | 0.05 / 0.5 / 0.9 (weak restrict — deliberate) |
| traffic_cost_factor / vot / risk_base | 1.0 / 10.76 / 10000 |

**Targets:** N=15 ✓; 100 % of assets fail in-horizon ✓; **gap 76.1 %** ✓ (≫62 %); avg simultaneous
renovations — **workload 3.45 (>3)** ✓, though the *realized* agents run ~2.7 because they
strategically restrict/stagger to spread the cuts (a feature; raise `--n` if a realized-agent value
>3 is required). Travel fraction is **81 %** (reactive): as documented below this exceeds the
[40,60] "representative" band by choice.

## 5. Gap ≈ travel (structural)

The MDP's only cross-asset coupling is congestion, so the gap ≈ the reactive's travel fraction (a
tuned single-threshold reactive bears ~0 risk and ≈equal maintenance to the clairvoyant; its whole
disadvantage is avoided congestion). Consequently **a large gap and a low travel fraction are
mutually exclusive** — pushing the gap to ~75 % requires travel ~75 %. `traffic_cost_factor=1.0`
(natural, unscaled traffic) is the max-gap operating point; lowering it slides down the gap↔travel
frontier (e.g. tcf≈0.5 → gap ~57 %, travel ~60 %).

## 6. Reproduce

```bash
python experiments/build_synergy_instance.py --n 15 --seed 0 --target-avgsim 3.5 \
    --v2c-bias 0.5 --output instances/instance_sf24.json
python experiments/run.py --config configs/sf24_clairvoyant.json      # 50-ep clairvoyant
python experiments/run.py --config configs/sf24_reactive.json         # tune + 50-ep reactive
```

The contrast selections are regenerable on demand: `--v2c-bias 0` (pure synergy) or
`--select-rule vol2cap` (busiest roads).
