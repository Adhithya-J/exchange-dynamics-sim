# TODO

Task list derived from the roadmap in
[`TRUST_SOCIETY_SCOPE.md`](TRUST_SOCIETY_SCOPE.md#implementation-order), expanded
with the current state of the code.

Current model: agents hold money (`resources`) and make direct transfers.
Optional layers add a goods market, trust beliefs and noisy reputation,
altruism/reciprocity, and environmental shocks with adaptation. A sweep runner
reports confidence intervals across seeds and parameters.

## 1. Define money, goods, utility, and conservation rules

- [x] Document what creates and destroys money and goods (production,
      consumption, living cost, transfers) in `MODEL.md`.
- [x] Decide whether `LIVING_COST` is a real payment (to producers) or an
      explicit sink; it remains an explicit money sink
      (`ResourceSimulation._apply_living_cost`).
- [x] Add config keys and validation for money/goods units and initial
      endowments (`MARKET` section and `_market_settings`).
- [x] Add tests asserting conservation holds except at documented sinks/sources.

**Done when:** every change in total money/goods is attributable to a named
rule, and tests enforce it.

## 2. Build a minimal market

- [x] Add goods inventory, production, and consumption to `Agent`.
- [x] Support producer/consumer roles or per-agent production/demand parameters.
- [x] Choose an initial price rule (fixed `MARKET.PRICE`).
- [x] Implement a step order: produce -> trade -> consume -> record.
- [x] Record inventory, unmet demand, and prices each iteration.
- [x] Add metrics for fulfilled demand, unmet demand, and trade volume.
- [x] Extend `config.yaml` and validation for the market settings.
- [x] Add tests for clearing, inventory leftovers, and zero-demand cases.

**Done when:** a run reports fulfilled/unmet demand and goods conservation,
with no regressions in the existing transfer tests.

## 3. Add private memory and noisy public reputation

- [x] Add decay-bearing private beliefs alongside the legacy `deque` of giver
      ids (`Agent.beliefs`).
- [x] Add a public reputation store shared by agents (`ResourceSimulation`).
- [x] Model observation noise, delay, and bias as configurable parameters.
- [x] Add metrics for reputation accuracy and calibration.
- [x] Add tests for memory decay and noisy reputation updates.

**Done when:** agents can hold an incorrect belief about another agent, and
the simulation reports how wrong those beliefs are.

## 4. Replace fixed probabilities with belief-based decisions

- [x] Base give/receive decisions on expected payoff and trust beliefs, blended
      with the legacy scores by `TRUST.DECISION_WEIGHT`.
- [x] Add risk tolerance held fixed per agent.
- [x] Keep decisions explainable (`decision_log` records the inputs used).
- [x] Add tests that a higher-trust belief increases cooperation.

**Done when:** decisions are driven by beliefs about other agents, not static
scores.

## 5. Add altruism and reciprocity as a separate social layer

- [x] Represent altruistic transfers with an explicit giver cost and recipient
      benefit.
- [x] Track reciprocity (who repaid whom, and how quickly) in `repayment_log`.
- [x] Keep this layer separate from market transactions.
- [x] Add metrics for altruism volume and reciprocity rates.
- [x] Add conservation tests for the social layer.

**Done when:** altruism can be enabled/disabled independently of the market and
is measured on its own terms.

## 6. Add environmental shocks and adaptation

- [x] Add a config-driven shock schedule (scarcity, information shocks).
- [x] Implement at least one adaptation/learning rule for agents.
- [x] Add resilience metrics (recovery time, survival after a shock).
- [x] Add deterministic-seed tests for shock runs.

**Done when:** a run can apply a shock at a chosen iteration and report how the
population recovers.

## 7. Run controlled experiments across seeds and parameter ranges

- [x] Add a sweep runner over seeds and parameter ranges.
- [x] Aggregate results with confidence intervals, not single runs.
- [x] Save results (CSV/JSON) for comparison.
- [x] Document the experiment workflow in `README.md`.

**Done when:** one command produces comparable results across seeds and
parameters.

## Cross-cutting

- [x] Emit a versioned, renderer-neutral replay log with typed events and
      explicit deaths (`replay.py`).
- [x] Keep the README model/limitations sections in sync with each change.
- [x] Extend `visualize.py` to chart new metrics (demand, reputation, shocks).
- [x] Add invariant/property tests for resource and goods conservation.
- [x] Add a CI workflow under `.github/` to run `pytest` on supported Python
      versions (3.11-3.14).
