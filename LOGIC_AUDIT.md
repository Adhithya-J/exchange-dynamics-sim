# Logic, Economic, Visual, and Game-Theory Audit

> **Status:** Sections 1 (1.1–1.8) and 2 (2.1–2.5) have been fixed and are
> covered by the test suite. The text below records the issues as they were
> found, not the current implementation.

**Scope:** `main.py`, `replay.py`, `experiments.py`, `visualize.py`, `viewer/`, `config.yaml`, `MODEL.md`, `README.md`, `TRUST_SOCIETY_SCOPE.md`.

**Method:** Code reading plus targeted runtime probes. The full test suite was run: **147 tests pass**. The issues below are therefore mostly semantic/design flaws and untested edge cases rather than failing assertions.

**Short version:** the model's stated units are inverted in practice. Money is the only thing that keeps agents alive, while goods have no utility. The trust layer does not measure or predict reciprocity, and several metrics are not measuring what their names and documentation claim.

---

## Table of Contents

1. [Critical Economic/Model Issues](#1-critical-economicmodel-issues)
2. [Game-Theory Inconsistencies](#2-game-theory-inconsistencies)
3. [Mathematical and Metric Bugs](#3-mathematical-and-metric-bugs)
4. [Visual and Presentation Logic](#4-visual-and-presentation-logic)
5. [Config Validation and Robustness Gaps](#5-config-validation-and-robustness-gaps)
6. [Documentation/Code Contradictions](#6-documentationcode-contradictions)
7. [Suggested Fix Order](#7-suggested-fix-order)

---

## 1. Critical Economic/Model Issues

### 1.1 Goods are economically inert; money is the real survival resource

- `Agent.is_alive` depends only on `resources` (`main.py:94-98`), and `_apply_living_cost` charges money (`main.py:1172-1189`). Goods never affect survival, productivity, or utility.
- `_run_market_cycle` produces, trades, and consumes goods (`main.py:1204-1290`), but the only consequence of consumption is reducing `unmet_demand` (`main.py:1271-1281`). `unmet_demand` only matters through `_adapt_to_scarcity` (`main.py:582-597`), and only when `SHOCKS.ENABLED` is true.
- `MODEL.md:9-15` says money is "not valuable on its own" and goods are "valuable only when consumed." In code, the opposite is true: money determines life/death; goods do nothing.
- Evidence: in a two-agent producer/consumer run, the consumer was forced to spend money on goods it could not use and died at iteration 50 instead of 100. Three agents with `unmet_demand = 5` each for 10 iterations remained fully alive with unchanged money.

**Consequence:** the market is not an economy. It is a forced money-transfer mechanism from net consumers to net producers, with goods as a decorative token. Under the model's own (implicit) preferences, buying is weakly dominated, yet the code forces it.

### 1.2 Extinction is structurally inevitable, and the default horizon hides it

- There is no money source. Living cost destroys money every iteration; transfers and trade only move it (`MODEL.md:19-30`).
- With default `LIVING_COST: 1.0` and `INITIAL_RESOURCES: 1000.0`, total money monotonically falls until everyone dies. For the checked-in seed 42 config, extinction occurs at **iteration ~1405**; the default is `ITERATIONS: 5000` (`config.yaml:5`).
- By iteration 5000, the default run's final metrics are:
  - `alive = 0`
  - `gini = 0.0`
  - `mean = 0.0`
  - `total = 0.0`
  - `unmet_demand = 0.0`
- Therefore `python experiments.py` with defaults aggregates mostly **zero-valued final metrics**. `main.py` also prints zero final statistics. The README example uses `--iterations 500`, which avoids this, but the default path is misleading.
- Gini rises to ~0.64 near collapse, then drops to 0 when everyone is dead. Inequality "improves" only through extinction.

### 1.3 The market is not really a market

- Price is fixed (`main.py:1220`), there is no supply response, no price discovery, and buyers/sellers have no strategic choice. Sellers always sell surplus; buyers always buy the shortfall up to the floor (`main.py:1228-1269`).
- The buyer may spend down to `RESOURCE_FLOOR` (`main.py:1232-1234`) with no reserve for the living cost charged later. The code therefore lets an agent spend money that it needs to survive on goods that do not help it survive.
- `MAX_RESOURCES_AFTER_RECEIVING` limits gift income but **not** market income (`main.py:1250-1254`), so trade can bypass the receiving ceiling.
- Because goods have no utility, the market's "equilibrium" is no trade; the simulation instead forces trade. Scarcity shocks reduce production of a good nobody needs, so they have little material effect when trust decisions are off.

### 1.4 The stated conservation invariant is false at the death boundary

- `MODEL.md:167-173` invariant 3 says total money changes only by `-LIVING_COST * living_agents`.
- In `_apply_living_cost`, an agent with less than `LIVING_COST` pays only what it has: resources are clamped to the floor (`main.py:1181-1186`). The actual removal is `before - after`, not `LIVING_COST`.
- Evidence: two agents with 0.5 each and cost 1.0 produce `living_cost_paid = 1.0`, not 2.0. Total money changes by `-1.0`, not `-2.0`.
- The replay test `test_replay_conserves_money_from_ticks_alone` reconciles against `living_cost_paid`, so it passes; but the documented invariant is wrong. The same boundary makes `death_log` cause categories ("gift", "repayment", "market") mostly unreachable under default parameters; nearly all deaths are `living_cost`.

### 1.5 Trust cannot learn trustworthiness

- `_observe_reciprocation` is called only from `_execute_transfer` on a **gift** (`main.py:1060`, `main.py:700-737`).
- Repayments never call it; `_process_repayments` only logs and moves money (`main.py:363-460`). Defaulting on a debt produces no negative observation.
- Public reputation is updated toward `1 + BIAS + noise` (`main.py:711-715`), not toward actual repayment behavior. With `REPAY_PROBABILITY: 0.0`, givers' reputations still rose to ~0.999 after 30 iterations — agents who never repay are indistinguishable from reliable ones.
- Repayment probability is a single global parameter (`config.yaml:65`), not an agent-specific trait. There is no hidden "trustworthiness" type, so reputation is not statistically predictive of repayment even in principle.
- The private belief updated is the **receiver's belief about the giver** ("this person gave me something"), not the giver's belief about whether the receiver will repay. The model measures gratitude, not trust in reciprocity.

**Consequence:** the central research question — "when does reputation improve cooperation?" — cannot be answered by this implementation, because reputation is not correlated with the outcome it is supposed to forecast. This directly contradicts `TRUST_SOCIETY_SCOPE.md:50-51` ("probability that this agent will reciprocate") and `MODEL.md:106-124`.

### 1.6 `RETURN_MULTIPLIER` is disconnected from the actual repayment rules

- The decision rule uses `expected_net = belief * RETURN_MULTIPLIER - 1` (`main.py:993-996`).
- Actual expected repayment per unit is `REPAY_PROBABILITY * REPAY_FRACTION`, default `0.5 * 1.0 = 0.5`, further truncated by the debtor's floor.
- With `belief = 0.5`, the decision model computes a break-even `2.0 * 0.5 - 1 = 0`, while the actual expected value is `0.5 - 1 = -0.5` per unit. With `belief = 0.9`, the model computes `+0.8`, but the actual expected value is still `-0.5`.
- If reciprocity is disabled but trust decisions are enabled, the decision layer optimizes a return that can never occur.

**Fix direction:** derive the multiplier from the social layer, e.g. `RETURN_MULTIPLIER = REPAY_PROBABILITY * REPAY_FRACTION`, or make repayment state-contingent and model it explicitly.

### 1.7 Acceptance of a gift is modeled irrationally

- `_receive_probability` uses the same expected-payoff logic as giving (`main.py:1009-1018`).
- With `SOCIAL.RECIPROCITY_ENABLED: false`, accepting a gift has zero cost. With defaults (`p=0.5`, `f=1.0`), accepting has expected net `+0.5 * amount`. A rational agent should accept any gift up to the receiving ceiling.
- The trust rule makes an agent *more* likely to accept from someone it trusts. If the concern is the future repayment obligation, the sign is backwards: you would be more cautious about accepting from someone you expect to hold you to the debt.
- Rejecting a free gift is therefore a dominated action, but the model can generate it.

### 1.8 Trust is counted multiple times in one decision

At `DECISION_WEIGHT = 1`, the same belief `p` is used:

1. in receiver selection: weight ∝ `0.01 + p` (`main.py:972-983`),
2. in the giving decision: `P(give) = p` (`main.py:998-1007`),
3. in the acceptance decision: `P(accept) = p` (`main.py:1009-1018`).

So the probability of a completed gift to a candidate is roughly proportional to `(0.01 + p) * p^2`, not the documented linear blend (`MODEL.md:128-140`). In a probe with `p_trusted = 0.982` and `p_distrusted = 0.018`, selection alone favored the trusted agent ~34.8×, and the give probability another ~54.6×. The effective trust effect is cubic, not linear. The test only checks selection in isolation (`tests/test_decisions.py:94-105`), so it does not catch this.

---

## 2. Game-Theory Inconsistencies

### 2.1 The reciprocity layer has no enforcement or punishment

- Debts are removed from the queue before affordability is checked (`main.py:388-395`). If the debtor is dead, broke, or rolls a failed repayment, the debt simply disappears.
- Evidence: after a debtor was reduced to 0, `_process_repayments` returned `outstanding_debt = 0.0` and `repayments = 0`. The metric named "outstanding debt" therefore under-reports true obligations.
- There is no consequence for default, no exclusion, no reputation hit, and no future refusal to lend. Cooperation cannot be sustained as a repeated-game equilibrium.

### 2.2 "Reciprocity rate" counts events, not value

- `total_repayments += 1` for every successful repayment (`main.py:437-441`), even if the debtor repaid only 1 of 10 because of the floor (`main.py:406-409`).
- `reciprocity_rate = total_repayments / total_gifts` (`main.py:447-451`) therefore overstates reciprocity. A partial refund counts the same as full repayment. No metric reports the fraction of gift value returned (`repayment_volume / altruism_volume`).

### 2.3 Memory is an always-on reciprocal channel, separate from the SOCIAL layer

- On receipt, `receiver.memory.append(giver.id)` (`main.py:1059`); receiver selection then prefers past givers via `_memory_weights` (`main.py:916-949`), with `MEMORY.PREFERENCE_FRACTION: 0.5` by default.
- This is already a return-favour mechanism. Enabling `SOCIAL.RECIPROCITY_ENABLED` adds a second, explicit debt for the same gift. The "altruism" metric therefore includes reciprocal behaviour, and the social layer double-counts reciprocity.

### 2.4 "Risk tolerance" behaves like decision noise, not risk aversion

- `_trust_decision_probability` is `sigmoid(expected_net / risk_tolerance)` (`main.py:989-996`). Higher "risk tolerance" flattens the logistic, moving every probability toward 0.5. For positive expected net, higher risk tolerance *reduces* the probability of giving; for negative expected net, it increases it.
- A risk-averse agent would require a larger expected payoff, not become indifferent. The parameter is closer to inverse rationality or decision temperature.
- Adaptation raises risk tolerance when demand is unmet (`main.py:582-597`), which flattens decisions — potentially reducing cooperation precisely under scarcity.

### 2.5 The model has no real strategic refusal/exploitation

- Agents cannot refuse to trade, cannot choose not to buy goods, cannot bargain, and cannot exploit a counterparty except by accepting a gift and defaulting (which has no consequence). The scope lists "trade, help, lend, refuse, or exploit" (`TRUST_SOCIETY_SCOPE.md:53-59`), but only a subset exists.

---

## 3. Mathematical and Metric Bugs

### 3.1 Reputation calibration metrics are invalid

- `_iteration_observations.append((predicted, 1.0))` (`main.py:717-718`): the Brier outcome is hard-coded to 1.0 for every successful gift. It never scores a default or a failed reciprocation.
- `reputation_brier` is therefore "how far is the giver's reputation from 1," not a forecast of reciprocation.
- `reputation_error` compares public reputation to a cumulative **giving rate** (`main.py:758-765`), not to repayment behavior or to the probability being predicted.
- Evidence: with `TRUST.BIAS = -1.0` (the model consistently observes 0), reputation correctly converged to 0.0, but `reputation_brier = 1.0` and `reputation_error = 1.0` — the worst possible scores. The metrics contradict the model's own observation.
- This directly contradicts `MODEL.md:121-124` and `TRUST_SOCIETY_SCOPE.md:70-79`.

### 3.2 `recovery_time()` is meaningless for information shocks

- It measures recovery of `unmet_demand`, which information shocks do not affect (`main.py:619-636`).
- An information shock at iteration 2 with duration 2 returned `recovery_time = 0`. The function should either use a shock-type-specific recovery variable or return `None`/N/A for information shocks.
- It also only considers the **last** shock in `shock_log`, so earlier shocks are ignored.

### 3.3 Adaptation runs even when there is no market or shock

- `_run_market_cycle` resets `_last_unmet_by_agent = {}` even when the market is disabled (`main.py:1208`). `_adapt_to_scarcity` then treats every agent as having met demand and lowers risk tolerance by `rate * 0.5` each iteration (`main.py:592-597`).
- Evidence: with `SHOCKS.ENABLED = true`, no scheduled shocks, and `MARKET.ENABLED = false`, risk tolerance fell from 0.5 to 0.25 after 10 iterations and to 0.0 after 20. Enabling an empty shock layer changes behaviour.
- It also drifts when the market is enabled but no shock occurs, because adaptation is unconditional. "Met demand lowers it" is applied every iteration, not only after a shock.

### 3.4 `RISK_TOLERANCE` is only drawn from config when trust decisions are enabled

- In `initialize_agents`, the range is used only if `trust["enabled"] and trust["decision_weight"] > 0` (`main.py:818`, `main.py:838-842`); otherwise risk is fixed at 0.5.
- If shocks are enabled but trust decisions are not, the configured `AGENTS.RISK_TOLERANCE` range is silently ignored.
- `MODEL.md:137-138` says risk tolerance is drawn once per agent from the config and stays fixed, contradicting the shock adaptation section (`MODEL.md:157-160`).

### 3.5 Experiment harness: zero iterations and confidence intervals

- `run_experiment(iterations=0, ...)` is allowed by the range check but crashes at `simulation.metrics_history[-1]` with `IndexError` (`experiments.py:126`).
- `summarize_runs` uses `1.96 * std / sqrt(n)` for every `n` (`experiments.py:176-188`). For `n = 2` or `3` — the README's recommended seed count — a normal 95% interval is far too narrow and can have zero width when `n = 1`. A t-interval or bootstrap would be appropriate.
- The default sweep metrics at 5000 iterations are all zero (see 1.2), so the default experiment summary is not informative.

### 3.6 Gini/mean/median include dead agents

- `calculate_statistics` uses all agents (`main.py:182-197`), including dead agents pinned at the floor.
- At extinction, Gini is 0 and mean is 0, which reads as "perfect equality" and "no wealth" rather than "everyone is dead." The viewer's "Gini 0.000" at `alive = 0` is especially misleading.
- Consider reporting alive-only statistics alongside all-agent statistics, or at least label them clearly.

### 3.7 Minor metric issues

- `mean_give_probability` and `mean_accept_probability` are 0.0 when trust decisions are off (`main.py:777-800`), even though gifts are happening. The names suggest they measure actual behaviour.
- `trust_observations` is per-iteration (`_iteration_observations` is reset each iteration), while `reputation_error` is cumulative. The metrics use different windows.
- `total_gifts` increments even when `SOCIAL.RECIPROCITY_ENABLED` is false (`main.py:349`), so the reciprocity denominator includes gifts that could never be repaid.
- `mean_risk_tolerance` includes dead agents (`main.py:599-611`), even though dead agents do not adapt.

---

## 4. Visual and Presentation Logic

### 4.1 "Commons" is total money, not a natural resource stock

- `stockRatio` is `total money / initial total money` (`viewer/viewer.js:325-327`; `replay.py:257-260`).
- The header shows both "Commons %" and "Coins"; they are the same series, one normalized. This is visually redundant and conceptually misleading.
- The help text says "The commons … is the village's shared natural stock" (`viewer/index.html:60-61`, `viewer/index.html:111`), while `MODEL.md:69-70` correctly says the meadow is a mirror of total money, not an extra resource stock.
- The commons does not respond to goods production or consumption at all. With the market enabled, goods can be abundant or depleted while the meadow is unchanged. A scarcity shock does not visibly thin the commons; only the money sink does.
- This is probably the single most misleading visual in the project: the village's "environment" is actually the money supply.

### 4.2 Coin boxes use a global, non-stationary scale

- `maxCoins` is the maximum balance over the **entire run** (`viewer/viewer.js:104-113`). If one agent later reaches a new maximum, every other agent's box shrinks visually even though their balance is unchanged. The same balance can appear to lose wealth over time.
- The box has only 12 slots (`viewer/viewer.js:1062-1065`), so it is a coarse quantized indicator rather than a proportional one.
- A fixed reference (initial resources, a percentile, or the current maximum) would make comparisons stable.

### 4.3 Dead agents' money and goods can be invisible but still counted

- `_apply_living_cost` sets dead agents' resources to the floor (`main.py:1183-1186`). If `RESOURCE_FLOOR > 0`, those coins remain in `total` and in the "Coins" statistic.
- `drawVillage` only draws huts and coin boxes for alive agents (`viewer/viewer.js:1144-1152`), and dead agents' goods are never drawn. The headcount drops, but the header can still show a positive coin total with no corresponding coin boxes.
- `MODEL.md:87-88` acknowledges stranded goods; the visual should either show estates or explain the discrepancy.

### 4.4 Replay event ordering is lost by the viewer

- `replay.py` sorts events by `(tick, phase)` and assigns a stable `seq` (`replay.py:339-353`).
- `replay-loader.js` then re-sorts by `(tick, type)` where gifts = 0, repayments = 1, trades = 2 (`viewer/replay-loader.js:103-145`). That puts all gifts before repayments before trades within a tick.
- The simulation's actual order is repayments → gifts → market (`MODEL.md:38-49`), and the replay's `seq` preserves it. The viewer discards `seq`, so the feed and animations show same-tick events out of order.
- `renderFeed` separately shows shocks, then deaths, then events (`viewer/viewer.js:1582-1650`), so a death caused by a gift can appear before the gift.

### 4.5 One position per agent per tick breaks multi-event animation

- `computeTargets` computes exactly one target position per agent per tick (`viewer/viewer.js:403-448`). Later events overwrite earlier ones.
- Example: A gives to B, and B gives to C in the same tick. B's position is set to C's hut by B's gift; the coin for A → B then ends at C's hut. Similarly, a gift followed by a trade moves the giver to the market for both animations.
- With 10 agents and several gifts per tick, this is common, not a corner case.
- The same mechanism means a giver or receiver who dies later in the tick is already drawn as a tombstone while their gift/trade coin is animating, because `aliveAt` uses the end-of-tick resource snapshot.

### 4.6 Death causes and shock stories are ignored

- The replay records causes (`gift`, `repayment`, `market`, `living_cost`) (`replay.py:67-72`), but the feed says only "ran out of coins" (`viewer/viewer.js:1598-1606`) and the story says "ran out of coins and left the village" (`viewer/viewer.js:1450-1453`). The cause is never shown.
- `describeTick` checks active shocks before events (`viewer/viewer.js:1454-1468`). During a multi-iteration shock, the story always describes the shock and hides all gifts/trades/repayments for the duration.
- The shock banner shows only the first active shock (`viewer/viewer.js:1511-1518`); overlapping shocks are not represented.
- The "quiet iteration" text says every living villager paid `cost * alive` (`viewer/viewer.js:1487-1490`), but at the death boundary some paid less than the full cost.

### 4.7 Bush/pond thresholds pop

- `drawCommonsBush` returns early when `ratio <= threshold` (`viewer/viewer.js:889-891`), so a bush disappears abruptly rather than thinning to zero.
- `drawCommonsPond` uses water `= base * (0.22 + 0.78 * level)` while above threshold, then 0 below (`viewer/viewer.js:950-960`). There is a visible discontinuity at the threshold.
- This is cosmetic, but it reads as a bug rather than a gradual depletion.

### 4.8 Capture script path bug

- README instructions (`README.md:168-176`) say `cd viewer` and run `node capture.mjs`.
- `capture.mjs:40` sets `viewerRoot = resolve(process.cwd(), "viewer")`. With cwd = `viewer/`, that becomes `viewer/viewer`, which does not exist. The server then 404s every `/viewer/...` request. Running `node viewer/capture.mjs` from the repo root works, but the documented invocation does not.
- Also, `--seconds 20` at 30 fps samples ~601 frames evenly across the whole replay, so each frame can jump ~2.3 ticks; it is a time-lapse sampler, not a smooth 30 fps animation.

### 4.9 Stale checked-in HTML artifacts

- `simulation.html` and `simulation-all-features.html` are from the older `const DATA`/self-contained viewer format; they contain no `window.__REPLAY__`, no `schema_version`, and no replay loader. Current `visualize.py` writes a replay-based viewer (`visualize.py:36-77`).
- The README says to open `simulation.html` (`README.md:201-213`), so the checked-in artifact is inconsistent with the current code. It may still render, but it is not generated by the current pipeline.

### 4.10 `Open replay` can exceed sessionStorage quota

- `bindFileInput` stores the entire replay as base64 in `sessionStorage` (`viewer/replay-loader.js:291-312`).
- The checked-in replay is ~2.25 MB uncompressed; base64 adds ~33%, so ~3 MB. That is within a typical 5 MB quota, but larger runs (more agents/iterations) will fail. IndexedDB or an object URL would be more robust.

---

## 5. Config Validation and Robustness Gaps

The engine validates some settings but not others. These can silently break the stated invariants:

- `LIVING_COST` is not validated (`config.yaml:25`, `main.py:1175`). A negative value creates money every iteration, breaking conservation.
- `INITIAL_RESOURCES` (`config.yaml:7`, `main.py:809`) is not validated. A negative value causes `calculate_gini` to raise "Gini is undefined for negative values" on the first iteration.
- `RESOURCE_FLOOR` (`config.yaml:9`) is not validated. A negative floor allows resources below zero via market/repayment, violating invariant 4.
- `MIN_RESOURCES_AFTER_GIVING` and `MAX_RESOURCES_AFTER_RECEIVING` (`config.yaml:21-23`) are not validated for range or ordering. A giving floor below the death floor can make a gift lethal; a receiving ceiling below initial resources silently disables gifts.
- `AGENTS.GENEROSITY` and `AGENTS.ACCEPTANCE` (`config.yaml:17-19`) are not validated as ascending or within `[0, 1]`. A reversed range silently collapses to a constant via `np.clip`.
- `MEMORY.SIZE` (`config.yaml:97`) can be negative, causing `deque(maxlen=-1)` to raise.
- `SIMULATION.AGENTS = 0` (`config.yaml:3`) leads to an empty `to_frame()` and a statistics error.
- `RISK_TOLERANCE` (`config.yaml:27`) is validated non-negative/ascending but not capped at 1, despite the documented `[0, 1]` range.
- `AFFORDABILITY.MIN_SCORE/MAX_SCORE` are not validated within `[0, 1]`, although they feed effective probabilities.

---

## 6. Documentation/Code Contradictions

- `MODEL.md:9-15`: money "not valuable on its own," goods "valuable only when consumed" — inverted in code.
- `MODEL.md:169`: total money changes by `-LIVING_COST * living_agents` — false at the bankruptcy boundary.
- `MODEL.md:121-124`: `reputation_brier` scores "observed reciprocation" — it uses a hard-coded outcome of 1.0 for gifts.
- `MODEL.md:137-138`: risk tolerance is drawn once and "stays fixed" — shocks adapt it; `MODEL.md:157-160` says so.
- `README.md:55-64`: "each agent produces `PRODUCTION_RANGE` goods per iteration" — the range is sampled once at initialization (`main.py:828-837`), not each iteration.
- `README.md:61-63`: "agents consume their own goods first" — the code trades before consumption; the outcome is equivalent, but the sequence is not.
- `config.yaml:8`: "RESOURCE_FLOOR: higher protects more wealth" — a higher floor raises the death threshold (agents die sooner) and strands more floor money in dead agents.
- `config.yaml:16-19`: "1 always tries" — affordability can reduce effective generosity/acceptance below 1.
- `config.yaml:26-27`: "higher [risk tolerance] means more willing to act on uncertain trust" — higher values flatten the logistic, making decisions more random, not more risk-tolerant.
- `TRUST_SOCIETY_SCOPE.md:50-51`: trust as "probability that this agent will reciprocate" — the code learns from gifts, not repayments.
- `TRUST_SOCIETY_SCOPE.md:36-37`: death must not be a clamp side effect — `_apply_living_cost` is exactly a clamp-based death.
- `viewer/index.html:60-61,111`: "Commons … shared natural stock" — it is total money.
- `README.md:94-97`: reputation/calibration claims — not supported by the metrics.

---

## 7. Suggested Fix Order

1. **Decide what goods do.** Either give consumption a survival/utility consequence (e.g., feed agents, affect productivity, or replace the money living cost with a goods cost) or explicitly label the market as a pure transfer/pedagogical toy.
2. **Make trust learn from reciprocity.** Add heterogeneous repayment propensities or state-contingent repayment, update reputation on repayment/default, and use repayment history rather than gift history. Then `RETURN_MULTIPLIER` can be derived from actual repayment parameters.
3. **Fix the decision model.** Remove the triple-counting of trust, make acceptance weakly dominant (or model the debt burden explicitly), and make risk tolerance mean risk aversion or rename it.
4. **Fix the experiment horizon and metrics.** Default to a horizon before extinction (or report extinction-relative metrics), exclude/report dead agents separately, and use t-intervals or bootstrap CIs.
5. **Fix the replay/viewer semantics.** Preserve `seq`, represent per-event positions, show death causes, and clarify that the commons is money (or make it track goods).
6. **Add validation/edge tests** for negative costs/floors, zero agents/iterations, reversed ranges, and the death-boundary conservation invariant.

---

## Reproduction Notes

Selected probes used during the audit:

- Default run (`seed=42`, 5000 iterations): extinction at ~1405; final `alive/gini/mean/total` all zero.
- Producer/consumer pair with `LIVING_COST=1`, `PRICE=1`: consumer dies at iteration 50 with the market on vs 100 with it off; producer survives to 150.
- Three agents with `unmet_demand=5` each for 10 iterations: all alive, money unchanged.
- `SHOCKS.ENABLED=true`, empty schedule, `MARKET.ENABLED=false`: risk tolerance 0.5 → 0.25 after 10 iterations → 0.0 after 20.
- Information shock at iteration 2: `recovery_time()` returns 0.
- `TRUST.ENABLED=true`, `SOCIAL.REPAY_PROBABILITY=0`: reputation rises to ~0.999 after 30 iterations.
- `TRUST.BIAS=-1`: reputation converges to 0.0, but `reputation_brier=1.0` and `reputation_error=1.0`.
- Broken debtor after a gift: `_process_repayments` returns `outstanding_debt=0.0`.
- Two agents with 0.5 money each and `LIVING_COST=1.0`: `living_cost_paid=1.0`, not 2.0.
- `run_experiment(iterations=0, ...)`: `IndexError`.
