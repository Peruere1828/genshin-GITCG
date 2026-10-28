---
title: "Strength-evaluation protocol (WS2)"
kind: canonical
status: valid
freshness: 2026-10-27
machines: [all]
authority: canonical
---

# Strength-evaluation protocol (WS2) — certifying an agent's level and gating training

> **2026-10-27 P0 update (D25)**: `--seed-start` no longer double-offsets task seeds;
> `match_seed` is recorded per row; `neural:` requires `--contestant-deck`.
> Power: 400 paired games = screening only; 800 detects ≥6pp; non-inferiority
> (`CI_low ≥ −0.02`) needs ~1,400 pairs; 3pp needs 2,000–3,000. See
> `docs/STATUS.md` §5 and `reports/explore/p0_parity_20261027.md`.

> Deliverable of exploration task 4. Prototype: `eval/scoreboard.py` (statistics),
> `scripts/eval_protocol.py` (runner), `tests/test_eval_protocol.py` (engine-free
> tests). Tiny-run evidence: `reports/explore/eval_protocol_smoke.json`
> (raw stream `data/eval/eval_protocol_smoke.jsonl`, gitignored).
> This document is the authority for the protocol; PLAN.md §2 (WS2), §4
> (M0/M2/M3-CPU/M4) and D11/D15 remain the authority for the acceptance *targets*.

The protocol answers two different questions with one scoreboard:

1. **Absolute level** — how strong is this agent against the fixed scripted pool?
   (M2 ≥ 0.50, M3-CPU ≥ 0.50, M4 ≥ 0.70 with no opponent < 0.45.)
2. **Real progress** — is a candidate better than the incumbent, or just luckier
   on the seeds? (training merge gate: "not worse than current checkpoint".)

Both must be measured **in pure-policy mode**; LLM-assisted mode is reported
separately (PLAN.md §6.3) and never feeds the strength gate.

---

## 1. Objects and notation

| Symbol | Meaning |
|---|---|
| `C` | contestant = (policy spec, deck slug). Policy spec via `envs.policy.build_policy`: `expert:<slug>` / `neural:<ckpt>[#t=<temp>]` / `heuristic` / `legal_random`. |
| `P = {o_1..o_K}` | fixed scripted pool, `K = 20` (`eval.opponents.scripted_opponents`), versioned (D11). |
| `S = {s_1..s_N}` | paired seeds, shared by every opponent and by every contestant. |
| `side ∈ {0,1}` | which player the contestant controls. **Both sides are played on the same seed** so first-player advantage cancels. |
| `x_{o,s,side}` | outcome score from the contestant's view: win `1`, draw `0.5`, loss `0`. |
| `n_o` | clean games vs opponent `o` (`= 2N` when both sides are played). |

A match is **clean** if the engine reported no error and the game did not exceed
`max_steps` (truncated). Only clean games enter win rates; error/truncated counts
are reported and gate the run (fallback/illegal rate must be `0`).

**Deck confound.** Policy strength is entangled with the piloted deck. A
certification therefore fixes *the deck set* and reports per-deck; do not mix
decks inside one pooled number unless the deck set itself is the deliverable.

---

## 2. Win-rate estimators

### 2.1 Per-opponent rate + Wilson interval

```
p_o = (Σ_s x_{o,s}) / n_o                       (draws already 0.5)
Wilson 95% CI (z = 1.96), p̂ = p_o, n = n_o:
  denom  = 1 + z²/n
  center = (p̂ + z²/(2n)) / denom
  margin = (z · sqrt( p̂(1−p̂)/n + z²/(4n²) )) / denom
  [lo, hi] = [center − margin, center + margin]  (clamped to [0,1])
```

Wilson is used instead of the normal approximation because our samples are small
and rates sit near the 0/1 edges (`eval/stats.py`).

### 2.2 Pooled rate — two intervals, only one is honest

```
pooled = (Σ_{o,s} x_{o,s}) / (Σ_o n_o)                      (game-weighted)
```

- **`pooled_ci_*` (Wilson over all games).** Optimistic: treats `K·N·2` games as
  exchangeable, ignoring that opponents are clusters (matchups differ
  systematically). Use it for display, not for the gate.
- **`cluster_ci_*` (opponent as the unit).** Compute per-opponent rates
  `p_o`, then treat `{p_o}` as `K` i.i.d. observations:

  ```
  cluster_mean = mean_o(p_o)
  SE           = sd_o(p_o) / sqrt(K)
  CI           = cluster_mean ± t(K−1) · SE          (t95 table in scoreboard.py)
  ```

  Plus a **cluster bootstrap** (resample the `K` opponents with replacement,
  `B=2000`, percentile 2.5/97.5) — reported as `cluster_boot_ci_*`. The bootstrap
  is the interval that should gate go/no-go decisions; it is wide when the
  opponent set is small (an honest reflection of "we only tested K matchups").

> Note on `K`: with `K=2` the t-interval is mathematically valid but useless
> (df 1 ⇒ `t=12.7`); the tiny run below shows `cluster95 = [−5.85, 6.85]`. This is
> a feature: **the protocol refuses to fake precision when the pool is small.**
> Use `K ≥ 8` for a meaningful cluster CI.

### 2.3 Paired-seed A/B difference

For two contestants `A`, `B` sharing seed keys `(o, s, side)`:

```
d_i = x^A_i − x^B_i                                (per shared match)
Δ   = mean_i(d_i)
s_d = sd_i(d_i),   SE = s_d / sqrt(n)
CI  = Δ ± z · SE
```

The paired design removes most deck/seed luck (when seeds are correlated,
`Var(d) = Var(A)+Var(B) − 2Cov(A,B)` is much smaller than the unpaired
`Var(A)+Var(B)`). This is the **training merge gate**: accept a checkpoint when
the paired CI lower bound is above a tolerance,

```
non-inferiority:  CI_low(Δ) ≥ −δ,   δ = 0.02  (default)
```

i.e. a one-sided non-inferiority test so seed noise cannot reject a genuinely
equal-or-better checkpoint. Report `Δ`, its CI, and per-opponent `Δ` as well.

### 2.4 "No single opponent below a floor" (M4)

The M4 rule reads "no single opponent win rate < 45%". Taken literally on point
estimates at small `N`, this is noisy and *will* fire by chance: an opponent
whose true rate is exactly 0.50 shows `< 0.45` with probability
`P(Binom(N,0.5) ≤ floor·N)`: `≈ 0.25` at `N=20`, `≈ 0.14` at `N=100`. With 20
opponents that is several spurious violations per run. The protocol therefore
splits the gate:

- **Screening (cheap, frequent):** pooled point estimate + per-opponent point
  estimate with `N ≈ 20–50`; a `< 0.45` opponent is flagged for *investigation*,
  not auto-fail.
- **Certification (rare, M2/M3-CPU/M4):** require the per-opponent Wilson
  **lower** bound `≥ 0.45`, and the pooled **cluster** CI lower bound `≥`
  threshold. Sample sizes in §5.

---

## 3. Ladder: fixed anchors + historical checkpoints

The pool is a **fixed ruler** whose ratings come from the L5.1 20×20 round-robin
(`reports/arena/20261008T163940_full_matrix_20x20.json`, anchors 1092–1875).
Two numbers are emitted:

### 3.1 Online Elo (existing, `eval/ladder.py`)

```
E_A       = 1 / (1 + 10^((R_B − R_A)/400))
R_A'      = R_A + K·(S_A − E_A),   K = 16,  S ∈ {1, 0.5, 0}
```

Order-dependent; kept for continuity with prior reports, not for certification.

### 3.2 Fixed-anchor Bradley-Terry / Elo MLE (certification)

Maximize the likelihood of the contestant's rating `r` against fixed anchors
`R_{o_i}` (same logistic scale as Elo):

```
p_i = σ((r − R_{o_i}) / S),   σ(u)=1/(1+e^{−u}),   S = 400/ln10 ≈ 173.72
logL(r) = Σ_i [ y_i·ln p_i + (1−y_i)·ln(1−p_i) ]
dlogL/dr = Σ_i (y_i − p_i) / S  = 0     (monotone ⇒ exact bisection)
```

CI by bootstrap over games (`B=2000`, percentile). This yields one comparable
scalar per checkpoint; because the anchors are fixed, `r` is comparable **across
runs and across time**. `historical checkpoints` become extra nodes: run the same
protocol for the old checkpoint and require

```
r_new − r_old  >  0    (see rating_difference)
```

in addition to the pool position. That is what makes "beat the old self" *not*
the same as "got stronger": a checkpoint can improve on the old self and still
sit below the pool bar, or vice versa.

> Anchor caveat: the L5.1 anchors are themselves an order-dependent online Elo,
> so `r` is a *relative* rating on a historical ruler, not an absolute skill
> number. Re-fit anchors (Bradley-Terry) if the ruler is ever rebuilt.

> Few-opponent clamp: with all-wins (or all-losses) against few anchors the MLE
> hits the `anchor ± 1500` boundary (tiny run: `3052.5`). Report it, and only
> trust `r` once the run covers a spread of opponents.

---

## 4. Decision-quality proxies

Not gates by themselves (except fallback/illegal = 0) but they localize *why* a
win rate moved and catch degenerate policies.

| Proxy | Formula | Notes |
|---|---|---|
| Fallback / illegal rate | `Σ fallbacks / Σ decisions` | `envs/match.py` counts an illegal/non-legal policy action as a fallback. **Must be exactly 0** for a valid run. |
| Error / truncation rate | `#(error or truncated) / #matches` | Must be 0; non-zero invalidates the run. |
| Round efficiency | `mean(rounds)`, split wins/losses | Faster wins and slower losses ⇒ stronger. |
| Decision efficiency | `Σ decisions / Σ rounds` | Fewer decisions per round at equal results ⇒ more decisive. |
| Draw rate | `#draws / #clean` | Engine draws treated as 0.5. |
| Matchup coverage | `#opponents with n_o>0 / K` | Certification requires full coverage. |
| **Search-value regret** | `regret_t = V_t(best) − V_t(chosen)` | Needs per-decision search values; report mean + p90. From `envs/fork_search` `SearchDecision` (plumbing TODO, §7). |
| **Teacher agreement** | `#(chosen == argmax teacher) / #decisions` | Needs a stronger teacher (search / CFR) on logged states; a monotone proxy when regret is unavailable. |

---

## 5. Required sample sizes (computed, reproducible)

Regenerate the tables with `python -m scripts.eval_protocol --sample-sizes`.

### 5.1 Pooled Wilson 95% CI half-width (p=0.5, K=20, wall-clock at L5.1 4900 games/h)

| target half-width | total games | games / opponent | hours @ 4900 g/h |
|---|---|---|---|
| 0.100 | 97 | 5 | 0.02 |
| 0.075 | 171 | 9 | 0.03 |
| 0.050 | 385 | 20 | 0.08 |
| 0.040 | 601 | 31 | 0.12 |
| 0.030 | 1068 | 54 | 0.22 |
| 0.020 | 2401 | 121 | 0.49 |
| 0.010 | 9604 | 481 | 1.96 |

(`n = z²·p(1−p)/h²`, `z=1.96`. The pooled number is optimistic—see §2.2.)

### 5.2 Games per opponent for the M4 floor (Wilson lower bound ≥ 0.45)

| true rate vs an opponent | games / opponent |
|---|---|
| 0.50 | 385 |
| 0.55 | 100 |
| 0.60 | 45 |
| 0.65 | 25 |
| 0.70 | 20 |

Certifying a floor you barely beat is expensive by construction. **Practical
target:** aim the weakest matchup at ≈ 0.55 and plan `N ≈ 100/opponent`
(≈ 2000 games, < 1 h at 12 workers) for the M4 certification run.

### 5.3 Detect an absolute change (unpaired, two-sided, 80% power)

| Δ | games / arm |
|---|---|
| 0.100 | 393 |
| 0.075 | 698 |
| 0.050 | 1570 |
| 0.030 | 4361 |
| 0.020 | 9812 |
| 0.010 | 39245 |

The **paired** design (§2.3) needs far fewer: `n = (z_{α/2}+z_β)²·σ_d²/Δ²`. With
seed correlation the per-pair variance `σ_d²` is typically ≪ the unpaired value,
which is why the merge gate uses paired seeds rather than two independent runs.

### 5.4 Recommended tiers

| tier | games | purpose | gate |
|---|---|---|---|
| smoke | 1–2 opp × 2–3 seeds × 2 sides | wiring / regression hook | clean only |
| screening | K=20 × 5 seeds × 2 sides (200) | "is it obviously working?" | pooled point ≥ 0.50, no error |
| **certification** | K=20 × 50 seeds × 2 sides (2000) | M2 / M3-CPU / M4 | cluster CI low ≥ bar, floor Wilson-low ≥ 0.45 |
| A/B merge | 20 × 20 paired, contestant vs incumbent | accept checkpoint | paired CI low ≥ −0.02 |

---

## 6. Exploitability / best-response probe

Exact exploitability is out of reach (PLAN.md §7 risk table). Two levels:

### 6.1 Cheap probe (implemented): fixed best-responder lower bound

Over the tested pool,

```
BR_lower(σ)   = 1 − min_o p_o        (max loss rate to a single fixed opponent)
matchup_gap   = mean_o p_o − min_o p_o
```

`BR_lower` is a **lower bound on exploitability**: some pure scripted opponent
captures that fraction. A high value means we are far from Nash on that
matchup. Track `min_o p_o` as the primary fragility signal.

### 6.2 Reduced-domain Nash-distance (designed, not built)

Define a reduced domain `D`: a few fixed decks + a restricted action subset
(e.g. one phase, or the top-5 abstract actions). On `D` the game is small enough
to build a payoff table / run tabular CFR, so we can compute

```
ε(σ) = max_β u(β, σ) + max_τ u(σ, τ)         (NashConv; zero-sum: one ≈ 0)
v*   = game value of D (from a converged CFR run)
Nash-distance ≈ u(BR(σ), σ) − v*  =  max_β u(β,σ) − v*
```

Both `max` terms are best responses *inside `D`* (enumerable / CFR-computable).
Report `ε(σ)` and `Nash-distance` as **separate, non-gating** research metrics;
only promote to a gate after the reduced domain is shown faithful to the full
game on known strategies.

---

## 7. Pure-policy vs LLM-assisted (fair comparison)

PLAN.md §6.3: the strength gate is **pure policy only**. Comparison rules:

1. **Same paired seeds, same pool, same sides** for both modes; run both and key
   by `(o, s, side)`.
2. Report the two modes **in separate columns/rows** (`reports/explore/`),
   never averaged together.
3. Primary LLM statistic is the **paired Δ(LLM − pure)** with its CI (§2.3) and
   the per-opponent breakdown; secondary are intervention rate, changed rate,
   failed/degraded rate, and latency (from `agents.llm_assist`).
4. LLM output is nondeterministic: repeat the same seed `k` times and report the
   spread; a single run cannot certify an LLM-assisted number.
5. LLM must always fall back to the pure policy on error/timeout; degraded games
   are flagged (mirrors `scripts/run_llm_matrix.py`).

Implementation status: the runner has `--mode llm` (LLM re-ranks a base proposer
via `agents.llm_assist`); the base is any `build_policy` spec, so "neural + LLM"
and "expert + LLM" are both expressible. Not run here (no key, and out of the
tiny-run budget).

---

## 8. Metric → gate map

| Gate | Metric | Rule (`scripts/eval_protocol.py` flag) |
|---|---|---|
| Valid run | clean | `fallback_rate == 0 and error_rate == 0` |
| Coverage | matchup coverage | all `K` opponents have `n_o > 0` |
| M2 / M3-CPU | pooled | `pooled ≥ 0.50` (screening) / cluster CI low ≥ 0.50 (`--strict-ci`) |
| M4 | pooled + floor | `pooled ≥ 0.70` and every opponent Wilson low ≥ 0.45 (`--pooled-threshold 0.70 --floor 0.45 --strict-ci`) |
| M4 vs M2 | ladder | `r_agent − r_M2 ≥ 0` (M4 ≥ 65% over M2 baseline) |
| Merge (WS5) | paired Δ vs incumbent | `CI_low(Δ) ≥ −0.02` and no opponent dropped below floor |
| Fragility | best-response | track `min_o p_o`; fail if `< floor` |
| Real progress | anchored rating + self | `r_new > r_old` **and** `r_new` clears the pool bar |

`--strict-ci` switches M2/M4 from the point-estimate screening form to the
cluster-CI certification form; run screening frequently and certify once per
milestone (or per opportunity-compute window).

---

## 9. Prototype: commands and tiny-run evidence

```bash
# planning tables (no engine)
python -m scripts.eval_protocol --sample-sizes

# tiny smoke: 2 opponents x 2 seeds x both sides = 8 matches (~20 s, 4 workers)
python -m scripts.eval_protocol \
  --contestant expert:superconduct_aggro \
  --opponents unyielding_geo natlan_battleship \
  --seeds 2 --workers 4 --tag eval_protocol_smoke

# training merge gate: paired A/B vs an incumbent checkpoint
python -m scripts.eval_protocol --contestant neural:<new>#t=0 \
  --baseline neural:<old>#t=0 --seeds 20 --workers 12

# full certification (M4 form)
python -m scripts.eval_protocol --contestant neural:<ckpt> \
  --seeds 50 --workers 12 --pooled-threshold 0.70 --strict-ci
```

Coverage of the runner: `expert:<slug>` / `neural:<ckpt>` (via `build_policy`,
same code path as the subprocess rollout), optional `--baseline` paired A/B,
`--mode llm`, streaming to `data/eval/<tag>.jsonl` with **resume by task_index**
and a spec-fingerprint sidecar, aggregate to `reports/explore/<tag>_<stamp>.json`
(+ stable `<tag>.json`).

**Tiny run result** (`reports/explore/eval_protocol_smoke.json`; contestant
`expert:superconduct_aggro`, both sides, seeds 0–1):

```
games = 8, pooled = 0.500  wilson95 [0.215,0.785]  cluster95 [-5.85,6.85]
  unyielding_geo     0.000 [0.000,0.490] n=4   <-- floor violation
  natlan_battleship  1.000 [0.510,1.000] n=4
quality: fallback_rate=0.0  error_rate=0.0  draw_rate=0.0  mean_rounds=8.9
anchored rating 1585.7 [1379.7,1791.6]   (vs L5.1 20x20 anchors)
best-response: worst=unyielding_geo rate=0.0 -> exploitability_lb=1.0
gates: pooled=PASS floor=FAIL clean=PASS -> FAIL
```

This is exactly the intended behaviour: the expert sits between the strongest
(0% vs `unyielding_geo`) and weakest (100% vs `natlan_battleship`) pool members,
the floor gate correctly fails, and the wide cluster CI is the honest price of a
2-opponent pool. The stream resumes as a no-op (one re-run re-aggregated 8/8
rows in ~1 s).

---

## 10. Limits (explicit)

- **Not i.i.d.** Games are clustered by opponent and correlated by seed; only the
  cluster CI / paired design are defensible. The naive pooled Wilson interval is
  optimistic and should not gate.
- **Small K.** The cluster CI is uninformative for `K < ~8` (tiny run: `K=2`).
- **Deck confound.** A policy number is only meaningful for a fixed deck (set).
- **Anchor drift.** The L5.1 anchors are a historical online Elo; treat the
  anchored rating as a relative ruler and re-fit if the ruler changes.
- **Exploitability not solved.** The best-response probe is a lower bound on a
  fixed pool; the reduced-domain Nash distance is designed but not implemented.
- **Search regret / teacher agreement are not wired yet** — the aggregation API
  is in place (`summarize` quality block) but per-decision search values are not
  flowing into the eval stream (see §4 TODO).
- **LLM mode** is nondeterministic and expensive; only a repeated-seed spread
  certifies it, and it never gates strength.
- **Version pinning.** Every run carries `engine.lock` (gitcg commit + card-pool
  version, D4); comparing runs across a card-pool migration is invalid by design.

---

## 11. Remaining work

1. Wire per-decision **search value regret** and **teacher agreement** into the
   eval stream (`envs/fork_search` `SearchDecision` → `data/` → `summarize`).
2. Build the **reduced-domain** solver and validate `ε(σ)` against a known
   baseline before it becomes a gate.
3. Add **historical checkpoints** as first-class ladder nodes (a small registry
   mapping name → policy spec) so `r_new > r_old` is a one-command check.
4. Promote cluster bootstrap to the primary CI once `K ≥ 8` is the norm, and add
   a Bradley-Terry refit of the anchors so the ruler is itself MLE-based.
5. CI hook: a `--smoke` screening gate on the last `main` checkpoint.
