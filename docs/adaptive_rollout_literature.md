# Statistically-Adaptive Rollout Budgeting for Action Selection in MDPs

**Context.** Our Monte-Carlo rollout agents (`agents/rollout.py`:
`MonteCarloRolloutAgent`, `SequentialMCRolloutAgent`, and `DCLAgent`) select an
action at each decision epoch by estimating `Q(s,a)` for a set of candidate
actions via forward simulation under a base heuristic, then taking the argmin
(minimisation of expected discounted cost). Today every candidate gets a
**fixed** number of rollouts (`n_rollouts`, default 30) before the comparison.
Build-time estimates on `instance_10p` show this is far too slow for the
24 h cluster walltime (`i10p_rollout_{empty,policy}` ≈ 5 days for 50 eval
episodes; sequential variants ≈ 40 h — see `EXPERIMENTS.md`, Exp 0).

The fixed budget is wasteful because **most candidate deviations are obviously
worse than the incumbent** and could be discarded after a handful of rollouts;
only near-ties need the full budget. This document surveys the methods that
formalise "spend rollouts where they matter" and recommends one for our
setting.

**Our setting — the constraints that pick the method.**

1. **Minimisation of expected discounted cost** — the comparison target is a mean
   (or median) over scenarios.
2. **Common random numbers (CRN), paired.** `rollout_noise()` keys env noise on
   `("rollout", seed, root_state, decision_t, rollout_idx)` and **excludes the
   candidate action**. So rollout index `r` gives *every* candidate at a decision
   epoch the **same scenario** → samples are **paired** across candidates. This is
   the single most important structural fact: it lets us test the *paired
   difference* `Δ_r = Q_challenger(r) − Q_incumbent(r)`, whose variance is far
   smaller than the marginal variance of either `Q`, because the shared scenario
   noise cancels.
3. **Heavy-tailed costs (CV ≈ 0.9).** Risk cost escalates with failure duration
   and maintenance costs are length-proportional → the per-scenario cost
   distribution is right-skewed. Tests that assume normality of the *raw* cost
   are mis-specified; we want either a CRN-paired difference (which is much
   better behaved than the raw cost) and/or a **nonparametric** rule.
4. **Local-search / sequential candidate structure.** `act()` does greedy local
   search over single-asset deviations: it repeatedly compares **one challenger
   against the current incumbent**. This is a *pairwise* incumbent-vs-challenger
   problem, not a flat "pick best of K simultaneously" problem — which steers us
   away from fixed-budget K-arm allocations and toward a **sequential pairwise
   elimination/stopping rule**.

---

## 1. Ranking & Selection (R&S) in simulation optimization

R&S is the simulation-optimization subfield devoted to exactly our question:
how many replications to give each of `k` competing system designs to identify
the best with a statistical guarantee.

### 1.1 Indifference-zone (IZ), fully-sequential: Kim–Nelson KN / KN++

- **Kim & Nelson (2001), "A fully sequential procedure for indifference-zone
  selection in simulation," *ACM TOMACS* 11(3):251–273.** The canonical
  fully-sequential IZ procedure ("KN"). It takes a first-stage sample to
  estimate variances, then proceeds one replication at a time, maintaining a set
  of *surviving* systems and **eliminating** any system whose running-sum
  cumulative difference from the best falls below a triangular continuation
  region. Guarantees probability of correct selection ≥ 1−α **whenever the true
  best is at least δ (the indifference zone) better** than the rest. δ is the
  smallest difference "worth detecting" — differences below δ are deemed not
  worth paying simulation for, which is exactly the early-stop behaviour we want.
- **KN++ (Kim & Nelson 2006; Hong & Nelson).** Updates the variance estimate as
  observations accrue (rather than freezing it after the first stage). Important
  caveat surfaced repeatedly in the literature: KN++'s asymptotic validity is
  proven **when there is no dependence across systems — i.e. no CRN**. The
  *original* KN handles CRN-induced positive correlation gracefully (CRN only
  *helps* IZ procedures by shrinking the variance of differences), but the
  variance-update variant assumes independence.
- **Weakness at scale.** KN's elimination thresholds come from a Bonferroni
  bound over the `k−1` pairwise comparisons; for large `k` this is conservative
  and over-samples. In our case `k` is small (≤ 3 deviations per asset per
  local-search step), so Bonferroni conservativeness is mild.

**Fit to us:** KN is a strong conceptual match — fully sequential, elimination-
based, CRN-friendly, with an explicit indifference zone δ that says "don't pay
to resolve differences smaller than δ." Its main cost is that it is *parametric*
(normal-difference assumption) and needs a δ in cost units, which is awkward to
set per decision epoch when cost scale varies.

### 1.2 Optimal Computing Budget Allocation (OCBA)

- **Chen, Lin, Yücesan & Chick (2000), "Simulation budget allocation for further
  enhancing the efficiency of ordinary computer simulation," *J. of Discrete
  Event Dynamic Systems* 10:251–270**; textbook: **Chen & Lee (2011),
  *Stochastic Simulation Optimization: An OCBA Approach*.** Bayesian/large-
  deviations framework: given a *total* budget `T`, allocate replications across
  designs to **maximise the Probability of Correct Selection (PCS)**. The optimal
  allocation gives more replications to designs with (a) smaller mean gap to the
  best and (b) larger variance — `N_i ∝ (σ_i/δ_{b,i})²` with the best design
  getting `∝ sqrt(Σ ...)`. Reported ~10× budget reduction vs. equal allocation
  for a target PCS.

**Fit to us:** OCBA is *budget-centric* (fixed total, maximise PCS) rather than
*confidence-centric* (stop when sure). It shines for a flat set of K designs
compared simultaneously, less so for incremental incumbent-vs-challenger local
search. It is also parametric (Gaussian posteriors). Useful as a fallback
allocation idea if we ever batch-compare all deviations of an asset at once, but
a poorer fit than a sequential pairwise rule for our local search.

---

## 2. Pure-exploration / best-arm identification (BAI) bandits

The ML-side reformulation of R&S: each "arm" is a candidate action, each "pull"
is one rollout; we want the best arm (lowest mean cost) with few pulls.

- **Maron & Moore (1993/1997), "Hoeffding Races," *NIPS 6*** (and *AI Review*
  1997). The origin of "racing": maintain a confidence interval (Hoeffding
  bound) around each model/arm's mean; **eliminate any arm whose lower bound
  exceeds the best arm's upper bound**. Survivors keep racing until one remains
  or the budget ends. Directly motivated by saving compute in model selection /
  leave-one-out — structurally identical to "stop rolling out clearly-worse
  actions."
- **Even-Dar, Mannor & Mansour (2002, 2006), "Action Elimination and Stopping
  Conditions…," *JMLR* 7:1079–1105.** Successive/median elimination with PAC
  `(ε,δ)` guarantees and matching sample-complexity bounds. The clean
  theoretical backbone of elimination racing.
- **Audibert, Bubeck & Munos (2010), "Best Arm Identification in Multi-Armed
  Bandits," *COLT*.** Fixed-budget view; introduces **Successive Rejects** (no
  confidence parameter needed) and the hardness measure `H = Σ 1/Δ_i²`.
- **Karnin, Koren & Somekh (2013), "Almost Optimal Exploration in Multi-Armed
  Bandits," *ICML*.** **Sequential Halving**: split the budget into
  `⌈log₂ k⌉` rounds; each round sample all survivors equally, then **drop the
  worst half**. Parameter-free, near-optimal simple regret, very simple to
  implement. Widely used inside MCTS at the root (see §4).
- **Kalyanakrishnan, Tewari, Auer & Stone (2012), "PAC Subset Selection in
  Stochastic Multi-Armed Bandits," *ICML*** — **LUCB**. Instead of eliminating,
  it repeatedly samples the two "most confusable" arms: the lowest-LCB arm among
  the current top-m and the highest-UCB arm among the rest, stopping when they
  separate. The fixed-confidence, *anytime* counterpart to elimination; its
  pairwise "sample the contested boundary" idea maps neatly onto incumbent-vs-
  challenger.
- **SPRT-based BAI** (e.g. recent "SPRT-based Efficient Best Arm Identification")
  ties the bandit stopping rule directly to the sequential test of §3.

**Fit to us:** The racing/elimination family is the right *mental model*
(eliminate clearly-worse candidates early). But the classic bounds (Hoeffding/
UCB) are built for **independent, bounded, marginal** rewards and **ignore the
CRN pairing** — applying them to our paired data throws away the variance
reduction that makes early stopping cheap. **Sequential Halving** is attractive
when we want to compare *all K deviations of an asset at once with a fixed total
budget and no parameters*; it is a good candidate for the per-asset batch view
of `SequentialMCRolloutAgent`. But it does not exploit CRN pairing either.

---

## 3. Sequential tests on paired CRN data (the best fit)

Because CRN makes our samples **paired**, the natural tools are sequential tests
on the per-scenario difference `Δ_r = Q_challenger(r) − Q_incumbent(r)`.

- **Wilcoxon signed-rank test (Wilcoxon 1945)** as a *stopping criterion.* The
  signed-rank test asks whether the paired differences `Δ_r` are symmetrically
  distributed about zero. It is **nonparametric** (robust to our heavy-tailed,
  skewed costs) and **paired** (exploits CRN). Used sequentially: after each
  added rollout, test `H0: median(Δ)=0` against the one-sided alternative
  "challenger better" (`Δ<0`) or "incumbent better" (`Δ>0`); stop and decide once
  the one-sided p-value drops below a threshold. This is precisely the rule
  Optuna's `WilcoxonPruner` implements (§5).
- **Sign test** — even weaker assumptions (only needs `P(Δ<0)`); cheaper but
  lower power than signed-rank. A reasonable ultra-robust fallback.
- **Wald's Sequential Probability Ratio Test (SPRT) (Wald 1945; Wald &
  Wolfowitz 1948, optimality).** The optimal sequential test between two simple
  hypotheses on the mean difference: accumulate the log-likelihood ratio of the
  `Δ_r`, stop when it crosses an upper (`(1−β)/α`) or lower (`β/(1−α)`)
  boundary; guarantees type-I/II error `≈ α,β` with **minimal expected sample
  size**. Applying SPRT to *differences* is the paired, CRN-aware analogue of
  KN. **Kleijnen & Shi (2021), "Sequential probability ratio tests: conservative
  and robust," *Simulation* 97(1)** specifically study SPRT on simulation output
  and note the *parametric* SPRT with estimated variance is anti-conservative,
  while Hall's modified SPRT is conservative — i.e. you must be careful with the
  variance estimate. SPRT needs a parametric likelihood (normal-Δ), which our
  skew makes shaky unless `n_startup` is large enough for a CLT on the
  difference.

**Fit to us:** A **sequential Wilcoxon signed-rank stop on the paired CRN
differences** is the cleanest match: nonparametric (handles CV≈0.9 skew), paired
(captures CRN variance reduction), pairwise (matches incumbent-vs-challenger
local search), and — decisively — **it is the exact rule we already validated
for Optuna heuristic tuning** (`experiments/optuna_heuristic_search.py`, the
`WilcoxonPruner` that pruned ~99.6 % of trials with no quality loss). Reusing it
keeps one statistical story across the codebase.

---

## 4. Use inside planning / MCTS

The planning community independently rediscovered "don't simulate clearly-worse
moves":

- **Progressive widening / unpruning** (Coulom 2007; Chaslot et al. 2008;
  Couëtoux et al. 2011 for continuous/double progressive widening). Restrict the
  number of actions considered at a node and relax it as visits grow `∝ N^α`.
  This is *candidate-set* control, complementary to *per-candidate budget*
  control.
- **Sequential Halving at the root — SHOT and "Trial-based Heuristic Tree
  Search."** Pepels, Cazenave, Winands & Lanctot (2014), "Minimizing simple
  regret in MCTS," and **Cazenave's SHOT**; **Anytime Sequential Halving in MCTS
  (Fabiano & others, 2024, arXiv:2411.07171)** removes the need to know the
  budget in advance. These replace UCT's cumulative-regret sampling at the root
  with a *simple-regret* (best-arm) allocation — exactly the "give few sims to
  bad root moves" idea.
- **Early cutoff / move pruning.** Discard moves whose preliminary value is far
  enough below the best (a racing rule inside the tree).
- **Hoeffding/UCB-based rollout cutoffs and adaptive simulation counts** appear
  throughout the MCTS-for-operations-research literature (e.g. Bertsimas et al.
  2014, "A comparison of MCTS and mathematical optimization for large-scale
  dynamic resource allocation").

**Fit to us:** Our `act()` is effectively a *one-ply* decision-time planner
(local search over the immediate joint action, value-to-go estimated by
rollouts). The MCTS lesson — *use a simple-regret/best-arm rule at the decision
node and cut clearly-worse candidates early* — is exactly what we are doing;
Sequential Halving is the MCTS-blessed parameter-free option if we prefer it
over a paired test.

---

## 5. The Optuna `WilcoxonPruner` as the design template

`experiments/optuna_heuristic_search.py` already uses
`optuna.pruners.WilcoxonPruner(p_threshold=0.1, n_startup_steps=5)` over a
**fixed CRN set of tuning episodes** (episode `k` is the same scenario across all
trials — the paired requirement). Its mechanism (from the Optuna 4.x source):

- It performs `scipy.stats.wilcoxon(diff_values, alternative=alt,
  zero_method="zsplit")` where
  `diff_values = current_trial_step_values − best_trial_step_values`, matched by
  **step id** (i.e. paired by the shared CRN instance index).
- For **minimisation**, `alt="greater"` (test "current is *worse* than best");
  it prunes when the one-sided **p-value `< p_threshold`** *and* a mean-
  consistency guard holds (`average_is_best` — don't prune if the current
  trial's mean is actually better than the best's). `zero_method="zsplit"`
  splits zero-differences evenly so exact ties don't bias the rank sum.
- `n_startup_steps` blocks pruning until at least that many paired observations
  exist (avoids deciding on 1–2 noisy points).
- `p_threshold` (default 0.1) is the aggressiveness knob: higher → prune
  sooner, more risk of dropping a true winner.

This is a *one-sided paired Wilcoxon signed-rank sequential stop with a startup
guard and a mean-consistency safety check* — directly transplantable from
"prune a hyper-parameter trial across instances" to "stop rolling out a
candidate action across scenarios."

**Mapping Optuna → rollout selection**

| Optuna pruning | Adaptive rollout |
|---|---|
| trial (hyper-params) | candidate action (challenger) |
| "best trial" so far | current incumbent action |
| intermediate value at step `k` = mean cost on tuning episode `k` | `Q_candidate(r)` on rollout scenario `r` |
| paired by **episode index** (fixed CRN set) | paired by **rollout index `r`** (CRN excludes action) |
| prune trial when sig. worse than best | stop & reject challenger when sig. worse than incumbent (and symmetrically, accept when sig. better) |
| `n_startup_steps=5` | `min_rollouts` |
| (study budget) | `max_rollouts` cap (= legacy `n_rollouts`) |
| `p_threshold=0.1` | `p_threshold` |

---

## 6. Recommendation

**Adopt a sequential, one-sided, paired Wilcoxon signed-rank stopping rule on the
CRN rollout differences, applied pairwise to incumbent-vs-challenger comparisons
in local search.** Concretely, for each `(incumbent, challenger)` comparison:

1. Draw rollouts on **shared CRN indices** `r = 0,1,2,…`, forming the paired
   difference `Δ_r = Q_challenger(r) − Q_incumbent(r)` (the immediate-cost part
   is deterministic and cancels nothing — it is just an offset; the rollout part
   shares CRN, so `Δ_r` is low-variance).
2. After `min_rollouts` (the `n_startup_steps` analogue), run the one-sided
   signed-rank test both directions:
   - if `P(Δ<0) < p_threshold` and `mean(Δ)<0` → **challenger wins**, stop;
   - if `P(Δ>0) < p_threshold` and `mean(Δ)>0` → **incumbent wins**, stop.
3. Otherwise add more rollouts up to `max_rollouts` (= legacy `n_rollouts`); at
   the cap, decide by the sign of the mean difference.

**Why this over the alternatives**

- **vs. KN / SPRT (parametric):** our costs are heavy-tailed (CV≈0.9); the
  nonparametric signed-rank test is robust without a variance model or an
  indifference-zone δ in cost units. SPRT is theoretically optimal but its
  validity hinges on a correct difference-variance estimate (Kleijnen & Shi warn
  it is anti-conservative otherwise) — more fragile here.
- **vs. OCBA / Sequential Halving (fixed-budget, K-arm):** our local search is
  *pairwise sequential* (incumbent vs. one challenger), so a confidence-based
  *stop*-rule fits better than a budget-*allocation* rule. Sequential Halving
  remains the recommended fallback if we later batch-compare all deviations of an
  asset simultaneously (natural for `SequentialMCRolloutAgent`).
- **vs. Hoeffding/UCB racing:** those ignore CRN pairing and bound the *marginal*
  reward; we would forfeit the variance reduction that makes early stopping cheap.
- **Decisive practical reason:** it is the **same test we already validated**
  with `WilcoxonPruner` for Optuna tuning (≈99.6 % pruning, no quality loss),
  giving one consistent, reviewer-friendly statistical story.

**Parameters** (mirroring Optuna): `p_threshold` (default 0.1), `min_rollouts`
(startup guard, default 5), `max_rollouts` (cap, default = legacy `n_rollouts`).
Keep `zero_method="zsplit"` and the mean-consistency guard so exact ties and
near-zero differences never trigger a spurious early decision. Determinism is
preserved because the CRN keying is unchanged and the test is a deterministic
function of the (deterministic) `Δ_r` sequence.

**Open caveats / future work.**
- Multiple-comparison inflation: local search runs many pairwise tests per epoch;
  each at `p_threshold` individually. As with Optuna's use, we accept this — the
  goal is *budget savings with high-probability agreement*, not a global
  correct-selection guarantee. If a guarantee is needed, wrap the per-comparison
  α with a Bonferroni/KN-style correction over the deviations considered.
- Signed-rank power at very small `n`: with `n=5` all-same-sign differences the
  exact one-sided p-value floor is `2⁻⁵≈0.031 < 0.1`, so a *clearly* dominant
  candidate is correctly resolved at the startup size — good. Set `min_rollouts`
  no lower than ~5 so the test has any power.
- For `SequentialMCRolloutAgent`'s per-asset "best of ≤4" structure, a
  Sequential-Halving root allocation is a clean parameter-free alternative worth
  prototyping later.

---

## References

- Kim, S.-H., & Nelson, B. L. (2001). A fully sequential procedure for
  indifference-zone selection in simulation. *ACM TOMACS*, 11(3), 251–273.
  https://dl.acm.org/doi/10.1145/502109.502111
- Kim, S.-H., & Nelson, B. L. (2006). On the asymptotic validity of fully
  sequential selection procedures (KN++). *Operations Research*.
- Frazier, P. I. (2014). A fully sequential elimination procedure for
  indifference-zone R&S with tight PCS bounds. *Operations Research*.
  https://people.orie.cornell.edu/pfrazier/pub/2011_Frazier.pdf
- Chen, C.-H., Lin, J., Yücesan, E., & Chick, S. E. (2000). Simulation budget
  allocation for further enhancing the efficiency of ordinary computer
  simulation (OCBA). *J. Discrete Event Dynamic Systems*, 10, 251–270.
  OCBA overview: https://mason.gmu.edu/~cchen9/ocba.html ;
  https://en.wikipedia.org/wiki/Optimal_computing_budget_allocation
- Chen, C.-H., & Lee, L. H. (2011). *Stochastic Simulation Optimization: An OCBA
  Approach.* World Scientific.
- Maron, O., & Moore, A. W. (1993/1997). Hoeffding Races / The Racing algorithm.
  *NIPS 6*; *Artificial Intelligence Review* 11.
- Even-Dar, E., Mannor, S., & Mansour, Y. (2006). Action elimination and
  stopping conditions for the MAB and RL problems. *JMLR*, 7, 1079–1105.
- Audibert, J.-Y., Bubeck, S., & Munos, R. (2010). Best Arm Identification in
  Multi-Armed Bandits. *COLT*. http://sbubeck.com/COLT10_ABM.pdf
- Karnin, Z., Koren, T., & Somekh, O. (2013). Almost Optimal Exploration in
  Multi-Armed Bandits (Sequential Halving). *ICML*.
  http://proceedings.mlr.press/v28/karnin13.pdf
- Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. (2012). PAC Subset
  Selection in Stochastic Multi-Armed Bandits (LUCB). *ICML*.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Ann. Math.
  Statist.* https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
- Wald, A., & Wolfowitz, J. (1948). Optimum character of the SPRT.
- Kleijnen, J. P. C., & Shi, W. (2021). Sequential probability ratio tests:
  conservative and robust. *Simulation*, 97(1).
  https://journals.sagepub.com/doi/10.1177/0037549720954916
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics
  Bulletin*, 1(6), 80–83.
- Couëtoux, A., Hoock, J.-B., Sokolovska, N., Teytaud, O., & Bonnard, N. (2011).
  Continuous upper confidence trees / double progressive widening. *LION*.
- Pepels, T., Cazenave, T., Winands, M. H. M., & Lanctot, M. (2014). Minimizing
  simple regret in Monte-Carlo Tree Search. *CGW/ICML workshop*; Cazenave, T.
  SHOT.
- Anytime Sequential Halving in Monte-Carlo Tree Search (2024).
  https://arxiv.org/pdf/2411.07171
- Optuna `WilcoxonPruner` docs:
  https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html
  ; tutorial:
  https://optuna.readthedocs.io/en/latest/tutorial/20_recipes/013_wilcoxon_pruner.html
