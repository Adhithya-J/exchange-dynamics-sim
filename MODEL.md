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

1. Social transfers between living agents.
2. Goods production, if the market is enabled.
3. Market trade for goods, if the market is enabled.
4. Consumption of goods, if the market is enabled.
5. Living cost for every living agent.
6. Metrics and agent snapshots.

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
