# DCL: faithful reproduction vs. the old hybrid

This note documents the rebuild of the DCL agent (`agents/dcl.py`,
`training/dcl_trainer.py`) into a faithful reproduction of **Deep Controlled
Learning** (Temizöz, Imdahl, Dijkman, Lamghari-Idrissi, van Jaarsveld,
*European Journal of Operational Research*, 2025; arXiv:2011.15122), and *where
and why* the previous implementation (preserved at `agents/dcl_old.py`) deviated.

## What published DCL is

Approximate policy iteration that casts control as **classification**. For `n`
rounds: (1) collect a fresh **on-policy** dataset under the current policy π_i
(warm-up L steps, then step forward and label each visited state with the
rollout-**improved** action found by a simulation oracle — **Sequential Halving +
Common Random Numbers** — with budget B_s = M·|A_s|); (2) train a neural-network
**classifier** from scratch on the (state, best-action) pairs → π_{i+1}; (3)
π_{i+1} becomes the base/rollout policy for the next round. Canonical DCL is
**value-function-free**; the deployed policy is just the classifier (cheap argmax).

## Where the old implementation deviated, and why

| # | Published DCL | `dcl_old.py` | Why it had diverged | Rebuild |
|---|---|---|---|---|
| 1 | Value-function-free | Learned a VFA and bootstrapped a **truncated** rollout with it | Full rollouts over `T=120` epochs are expensive; truncation + VFA cut cost | **Opt-in** (`rollout_horizon=K`). Default `null` ⇒ full rollout, no VFA (faithful) |
| 2 | Sequential Halving + CRN over `A_s` | Fixed-`n_rollouts` local search; no SH, no early stop | The joint action space is `4^N` — textbook SH over `\|A_s\|` is infeasible, so local search was used | `rollout_selection` ∈ {`fixed` (default), `wilcoxon`, `sequential_halving`}; SH runs over the local-search **neighbourhood** (see below) |
| 3 | Batched on-policy rounds; classifier retrained from scratch | Online replay-buffer refit on a batch **mixing labels from many past policies** | Shoehorned into the generic value-based `Trainer` | `DCLTrainer` runs the `n`-round loop with a fresh on-policy dataset each round |
| 4 | Deployed policy = classifier | `act()` ran rollout local search **at eval too** | "Label generation" conflated with "acting" | `act()` = classifier inference; the oracle only labels during collection |
| 5 | Base policy π_i seeds sampling + rollout continuation | Warm-start commented out; heuristic→learned switch keyed off batch size and re-printed every update | Half-finished edits; the switch compared the wrong quantity | Structural: each round rebuilds the base/rollout policy = current classifier |
| 6 | — | `action_gen`/`_n_updates` dead; resume didn't persist the switch | Leftover scaffolding | Removed; `DCLTrainer` checkpoints `{round, classifier, VFA}` and resumes |
| 7 | Classifier trained to convergence each round | `NNPolicy.fit` did **one** gradient step | Built for the online loop | `_MLPEstimator` trains multi-epoch with frozen input normalisation |

## The one necessary adaptation: SH over a combinatorial action space

Published DCL applies Sequential Halving to the enumerable action set `A_s`. Here
the **joint** action is `(a_1,…,a_N) ∈ {none,repair,renovate,restrict}^N`, i.e.
`4^N` — not enumerable. The rebuild keeps SH+CRN as the literal best-arm oracle but
runs it over a **tractable candidate set**, via one of three `action_search`
decompositions:

- **`sequential`** — expanded MDP: decide assets in index order; exogenous
  information arrives only after the last asset. The oracle
  (`SequentialMCRolloutAgent`) fixes committed assets, completes the rest with the
  base policy inside each rollout, and SH chooses asset *i*'s action over ≤4 arms.
  The classifier is conditioned on the **partial post-decision state** (committed
  prefix applied), so it anticipates how later assets are filled in — the "PV
  anticipates the fill-in" requirement. This is the recommended/primary variant.
- **`independent`** — each asset predicted independently (plain per-asset
  classifier) plus a feasibility **coordinator**; labels come from joint local
  search. Cheapest, weakest at coordination.
- **`local_search`** — an autoregressive **edit policy** over a `3N+1` token space
  (set asset *i* to {repair,renovate,restrict}, or STOP). Labels are the canonical
  edit sequence reaching the oracle's joint action. Most expressive head.

CRN is supplied by `rollout_noise` (`agents/rollout.py`), whose key **excludes the
candidate action**, so all arms in an SH round replay identical exogenous
scenarios (positive covariance ⇒ variance reduction), exactly as the paper
prescribes.

## Optional truncated-rollout VFA (compute shortcut)

When `rollout_horizon=K` is set, each rollout runs K steps and adds
`γ^(K+1)·V(s_trunc)` at the truncation epoch, where `V` is a pre-decision
cost-to-go VFA fit each round on the n-step return-to-go of the on-policy data.
This trades a little bias for a large speed-up on the heavy per-step TAP
transition. With `rollout_horizon=null` the term is never added and the method is
value-function-free, as in the paper. The VFA lives in `DCLTrainer` (passed to the
oracle); it is **not** part of the deployed `DCLAgent`.
