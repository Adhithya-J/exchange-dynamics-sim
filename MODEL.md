# Model Rules and Conservation

This document is the source of truth for what enters, leaves, and moves inside
the simulation. Update it whenever a rule changes, and keep the conservation
tests in `tests/test_conservation.py` in sync.

## Units

| Unit | Stored on | Meaning |
| --- | --- | --- |
| Money | `Agent.resources` | Medium of exchange. Not valuable on its own. |
| Goods | `Agent.goods` | Consumable output. Valuable only when consumed. |

Both are non-negative floats. An agent is dead when its money is at or below
`SIMULATION.RESOURCE_FLOOR`.

## Conservation ledger

| Process | Money | Goods | Notes |
| --- | --- | --- | --- |
| Social transfer | conserved | unchanged | The giver loses exactly what the receiver gains. |
| Production | unchanged | **source** | Living agents add `production_rate` goods. |
| Market trade | conserved | conserved | The buyer pays the seller; goods move with the payment. |
| Consumption | unchanged | **sink** | Living agents destroy up to `consumption_need` goods. |
| Living cost | **sink** | unchanged | Each living agent is charged `LIVING_COST` money. |

Every change in total money or goods must be attributable to one row above.
Production is the only goods source, consumption is the only goods sink, and
living cost is currently the only money sink. Trade and altruistic transfers
only move value between agents.

## Death rule

- Death is explicit: `resources <= RESOURCE_FLOOR`.
- Dead agents neither produce, trade, consume, nor pay living cost.
- Reaching the floor does not top the agent up (`main.py:_apply_living_cost`).

## Iteration order

1. Expire active shocks and activate scheduled ones.
2. Decay trust and deliver delayed reputation observations.
3. Pay due reciprocal repayments.
4. Social transfers between living agents.
5. Goods production, if the market is enabled.
6. Market trade for goods, if the market is enabled.
7. Consumption of goods, if the market is enabled.
8. Living cost for every living agent.
9. Adaptation of risk tolerance.
10. Metrics and agent snapshots.

## Replay logs

`transfer_log` records every completed gift and repayment (iteration, giver,
receiver, amount, and kind). `trade_log` records every market trade (iteration,
buyer, seller, quantity, price, and money). `death_log` records every death and
its cause (`gift`, `repayment`, `market`, or `living_cost`). They are
append-only observations written after a step has been decided, so they cannot
change the simulation. `agent_history` stores one row per agent and iteration,
including resources, goods, risk tolerance, alive status, and public reputation
when enabled. The `living_cost_paid` metric records the money removed by the
living-cost sink each iteration, so total money can be reconciled from the logs.

`replay.py` packages those observations into a versioned JSON replay
(`schema_version`, `meta`, `agents`, `ticks`, and typed `events`). The replay is
the contract between the model and a renderer; it records state and events, not
positions or drawing details. The world visualization reads the replay; huts,
paths, coin boxes, and terrain are a presentation layer only, not part of the
model. Its outer meadow is a visual mirror of total money (`series.total`), not
an extra resource stock or a second ledger; the bushes and ponds thin only when
the recorded total falls, which in the default rules means the living-cost sink.

## Minimal market rules

The market is optional and disabled by default; with `MARKET.ENABLED: false`
the simulation behaves exactly like the transfer-only model.

- Every living agent produces `production_rate` goods per iteration. The rate
  is drawn once at initialization from `MARKET.PRODUCTION_RANGE`.
- Every living agent needs `consumption_need` goods per iteration, drawn from
  `MARKET.CONSUMPTION_RANGE`.
- Agents first consume their own goods. A shortfall is bought from other living
  agents' surplus, defined as `goods - consumption_need`, so sellers keep
  enough to cover their own need.
- The price is fixed at `MARKET.PRICE` money per good (no price discovery yet).
- A buyer may spend down to `SIMULATION.RESOURCE_FLOOR` but not below it.
- Unsold goods stay in inventory. Unmet demand is recorded.
- Goods held by dead agents are stranded but still counted in total goods.

## Altruism and reciprocity (optional)

Every transfer between agents is an altruistic gift: the giver pays an explicit
cost and the receiver gains the same amount. Market trades are separate and
never create gift debts.

When `SOCIAL.RECIPROCITY_ENABLED` is true, each gift is recorded as a debt from
the receiver to the giver. After `REPAY_DELAY` iterations the debtor attempts a
repayment with probability `REPAY_PROBABILITY`, returning `REPAY_FRACTION` of
the gift while staying at or above `RESOURCE_FLOOR`. Repayments are ordinary
money transfers, so they conserve total money.

`repayment_log` records the debtor, creditor, original gift iteration,
repayment iteration, delay, and amount. Metrics include `altruism_volume`,
`repayments`, `repayment_volume`, `reciprocity_rate`, `outstanding_debt`, and
`mean_repayment_delay`.

## Trust and reputation (optional)

Disabled by default. When `TRUST.ENABLED` is true:

- every agent starts with public reputation and private beliefs equal to
  `INITIAL_BELIEF`;
- after a successful transfer, the receiver observes the giver as cooperative;
- the private belief moves toward 1 by `LEARNING_RATE`, then blends with public
  reputation by `PUBLIC_WEIGHT`;
- public reputation moves toward the observed value (1 plus `BIAS` and Gaussian
  `NOISE`, clipped to [0, 1]);
- observations reach public reputation after `DELAY` iterations;
- beliefs and reputation are pulled back toward the prior by `DECAY` each
  iteration.

Trust only records and scores behaviour; it does not move money or goods, so the
conservation ledger is unchanged. `reputation_error` compares public reputation
with each agent's observed giving rate, while `reputation_brier` scores this
iteration's predictions against observed reciprocation.

### Belief-based decisions

When `TRUST.DECISION_WEIGHT` is greater than 0, giving and accepting blend the
legacy affordability scores with a trust decision:

- expected net payoff = `belief * RETURN_MULTIPLIER - 1`;
- trust probability = logistic(`expected net payoff / risk_tolerance`);
- final probability = `(1 - DECISION_WEIGHT) * legacy + DECISION_WEIGHT * trust`;
- receiver selection multiplies each candidate's memory weight by the same
  trust factor, so trusted agents are approached more often.

`RISK_TOLERANCE` is drawn once per agent from `AGENTS.RISK_TOLERANCE` and stays
fixed. Each attempted transfer is recorded in `decision_log` with the belief,
risk tolerance, probabilities, offered amount, and outcome, so decisions remain
explainable. Metrics report the mean give/accept probabilities.

The decision layer only changes which transfers happen; it never creates or
destroys money or goods.

## Environmental shocks and adaptation (optional)

Disabled by default. When `SHOCKS.ENABLED` is true, events in
`SHOCKS.SCHEDULE` activate at their 1-based `iteration` for `duration` steps:

- `scarcity`: multiplies goods production by `max(0, 1 - magnitude)`;
- `information`: adds `magnitude` to reputation observation noise.

Scarcity changes the goods source and information changes observations only;
neither directly creates or destroys money or goods. `shock_log` records every
activation.

At the end of each iteration, living agents adapt: unmet demand raises
`risk_tolerance` by `ADAPTATION_RATE` (capped at 1), while met demand lowers it
by half that rate (floored at 0). Metrics report `shock_active`, `shock_count`,
`production_modifier`, `reputation_noise`, `mean_risk_tolerance`,
`iterations_since_shock`, and `alive`. `recovery_time()` returns the number of
iterations from the last shock until unmet demand returns to its pre-shock
level, or `None` if it never does.

## Invariants

1. A social transfer or market trade never changes total money.
2. Total goods change only by `produced - consumed`.
3. Total money changes only by `-LIVING_COST * living_agents`.
4. No agent's money becomes negative.
5. Dead agents do not participate in any step.
6. Trust beliefs and public reputation always stay within [0, 1].
7. The trust layer never changes money or goods totals.

## Open decisions

- Replace the living-cost money sink with paid consumption once prices react to
  scarcity.
- Decide whether stranded goods pass to other agents when an owner dies.
- Add price discovery to replace the fixed `MARKET.PRICE`.
- Decide whether private beliefs should fully replace the legacy memory deque.
