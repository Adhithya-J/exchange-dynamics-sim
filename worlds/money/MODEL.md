# Model Rules and Conservation

This document is the source of truth for the money world in
`worlds/money/`. Update it whenever a rule changes, and keep the conservation
tests in `tests/test_conservation.py` in sync.

## Units

| Unit | Stored on | Meaning |
| --- | --- | --- |
| Money | `Agent.resources` | Medium of exchange, unit of account, and store of value. |
| Goods | `Agent.goods` | Consumable output. Consuming a good avoids an emergency money cost. |
| Commons goods | `ResourceSimulation.commons_goods` | Shared natural stock from which agents can extract goods. |
| Commons fund | `ResourceSimulation.commons_fund` | Money received as subsistence payments and fines. It is never destroyed. |
| Repayment probability | `Agent.repayment_probability` | Hidden trait: the chance this agent repays a due gift. |

Money and goods are non-negative floats. An agent is dead when its money is at
or below `SIMULATION.RESOURCE_FLOOR`.

## Money and its four functions

- **Medium of exchange:** gifts, repayments, and market trades all use money.
- **Unit of account:** prices, gifts, debts, and fines are all quoted in money.
- **Store of value:** money is conserved. Subsistence payments move to the
  commons fund and stay there until `FUND_REDISTRIBUTION` returns part of the
  fund to living agents.
- **Standard of deferred payment:** a gift can create a money debt that is due
  after `REPAY_DELAY`; the debt survives default, grows by
  `DEFAULT_PENALTY_RATE`, and remains payable later.

Money has no direct utility. It is valuable because it buys goods and because
it is accepted for taxes, rent, gifts, and debt settlement. Its real purchasing
power changes with the goods price and the state of the commons, as it would in
a real economy.

## What makes goods valuable

An agent needs `consumption_need` goods per iteration. Goods it owns are
consumed first. Any remaining unmet demand is charged in money at the
living-cost step at `MARKET.UNMET_DEMAND_PENALTY` per unit. A good is therefore
worth up to that penalty to a buyer: buying a good at a price below the penalty
reduces the money the agent will lose to unmet demand.

When the market price is high, an agent can instead extract goods from the
commons. Extraction is less efficient than buying: `EXTRACTION_EFFICIENCY` of
each raw unit becomes usable goods and the rest is waste. Each agent is limited
to `EXTRACTION_LIMIT` usable goods per iteration. The commons regenerates
`REGENERATION` raw goods per iteration up to `CAPACITY`.

A rational buyer compares the market price with `EXTRACTION_COST`, the
money-equivalent effort of extracting one good:

- if `price <= EXTRACTION_COST`, buy first and extract only what the market
  cannot supply;
- if `price > EXTRACTION_COST`, extract first and buy only the remaining
  shortfall;
- if `price > UNMET_DEMAND_PENALTY`, do not buy at all.

With `MARKET.ENABLED: false` every consumption need is zero, so no extraction
occurs and the simulation is the original transfer-only model.

## Conservation ledger

| Process | Money | Goods | Notes |
| --- | --- | --- | --- |
| Social transfer | conserved | unchanged | The giver loses exactly what the receiver gains. |
| Repayment | conserved | unchanged | A debtor returns part of an earlier gift. |
| Production | unchanged | **source** | Living agents add `production_rate * production_modifier` goods. |
| Commons regeneration | unchanged | **source** | The commons adds up to `REGENERATION` raw goods. |
| Market trade | conserved | conserved | The buyer pays the seller; goods move with the payment. |
| Extraction | unchanged | **moves + sink** | Usable goods move from commons to agent; the rest is waste (`commons_waste`). |
| Consumption | unchanged | **sink** | Living agents destroy up to `consumption_need` goods. |
| Subsistence payment | **transfer** | unchanged | Each living agent pays `LIVING_COST + UNMET_DEMAND_PENALTY * unmet_demand` into `commons_fund`. |
| Fund dividend | **transfer** | unchanged | `FUND_REDISTRIBUTION` of the fund is paid equally to living agents. |

Every change in total money or goods must be attributable to one row above.
Production and regeneration are the goods sources; consumption and extraction
waste are the goods sinks. Subsistence payments and dividends only move money
between agents and the fund. The charge is capped at what the agent can pay
while staying at or above `RESOURCE_FLOOR`, so the exact money moved is recorded
as `living_cost_paid`; `living_cost_due` records the nominal charge.

## Death rule

- Death is explicit: `resources <= RESOURCE_FLOOR`.
- Dead agents neither produce, trade, consume, nor pay subsistence costs.
- Reaching the floor does not top the agent up
  (`worlds/money/world.py:_apply_living_cost`).
- A death during the subsistence step is logged with cause `living_cost`; the
  cause covers both the fixed cost and the unmet-demand penalty.

Money is conserved, not destroyed, but agents can still run out of it: payments
accumulate in the commons fund unless `FUND_REDISTRIBUTION` returns them. Every
run can therefore still reach agent extinction if it runs long enough.
`ITERATIONS` defaults to 1000, which is below the extinction point for the
checked-in seed and keeps final metrics meaningful. The metric
`extinction_iteration` records the first iteration at which no agent is alive
(`null` before then). At extinction the fund may still hold money, so
`total_money` can remain positive while `total` (agent money) is zero.

## Iteration order

1. Expire active shocks and activate scheduled ones.
2. Decay trust and deliver delayed reputation observations.
3. Pay due reciprocal repayments and update trust from the outcome.
4. Social transfers between living agents.
5. Commons regeneration, goods production, market trade, and extraction, if the
   market is enabled.
6. Consumption of goods, if the market is enabled.
7. Subsistence cost for every living agent, paid into the commons fund.
8. Dividend from the commons fund to living agents.
9. Adaptation of risk tolerance.
10. Metrics and agent snapshots.

## Replay logs

`transfer_log` records every completed gift and repayment (iteration, giver,
receiver, amount, and kind). `trade_log` records every market trade (iteration,
buyer, seller, quantity, price, and money). `death_log` records every death and
its cause (`gift`, `repayment`, `market`, or `living_cost`). They are append-only
observations written after a step has been decided, so they cannot change the
simulation. `agent_history` stores one row per agent and iteration, including
resources, goods, risk tolerance, alive status, and public reputation when
enabled. The `living_cost_paid` metric records money moved into the commons fund
each iteration. `total_money = total + commons_fund` is conserved, and
`goods_total_all = goods_total + commons_goods` reconciles goods.

`worlds/money/replay.py` packages those observations into a versioned JSON replay
(`schema_version`, `meta`, `agents`, `ticks`, and typed `events`). The replay is
the contract between the model and a renderer; it records state and events, not
positions or drawing details. The world visualization reads the replay; huts,
paths, coin boxes, and terrain are a presentation layer only, not part of the
model. Its outer meadow is driven by `commons_ratio`, the remaining natural
goods stock, while the coin boxes and header money total use `stock_ratio`
(total money including the fund).

## Minimal market rules

The market is optional and disabled by default; with `MARKET.ENABLED: false`
the simulation behaves exactly like the transfer-only model.

- Every living agent produces `production_rate` goods per iteration. The rate is
  drawn once at initialization from `MARKET.PRODUCTION_RANGE`.
- Every living agent needs `consumption_need` goods per iteration, drawn from
  `MARKET.CONSUMPTION_RANGE`.
- Agents consume their own goods first. A shortfall is bought from other living
  agents' surplus, defined as `goods - consumption_need`, so sellers keep enough
  to cover their own need.
- A buyer will not pay more than `MARKET.UNMET_DEMAND_PENALTY` per good. Above
  that price the emergency cost of unmet demand is cheaper than the good, so
  the buyer refuses to trade.
- A buyer keeps `RESOURCE_FLOOR + LIVING_COST` in money and spends only the
  remainder, so trade cannot consume the money needed for the fixed subsistence
  cost.
- A seller's market income is capped by `MAX_RESOURCES_AFTER_RECEIVING`, the
  same ceiling that applies to gifts. A seller with `SELLER_RESERVATION_PRICE`
  above the market price holds goods instead of selling.
- A buyer with unmet demand can extract from the commons. Extraction is limited
  by `EXTRACTION_LIMIT`, loses `1 - EXTRACTION_EFFICIENCY` of each raw unit, and
  is preferred over buying only when the price is above `EXTRACTION_COST`.
- The price starts at `MARKET.PRICE` and adjusts once per iteration:
  `imbalance = (unmet_demand - leftover_goods) / total_demand`, then
  `next_price = price * (1 + PRICE_ADJUSTMENT * imbalance)`, clamped to
  `MIN_PRICE` and `MAX_PRICE`. `PRICE_ADJUSTMENT: 0` keeps a fixed price.
- Unsold goods stay in inventory. Unmet demand is recorded.
- Goods held by dead agents are stranded but still counted in total goods.

## Altruism and reciprocity (optional)

Direct transfers are gated by `GIFTING.ENABLED`, which is false by default. When
it is true, every transfer between agents is an altruistic gift: the giver pays
an explicit cost and the receiver gains the same amount. Market trades are
separate and never create gift debts.

When `SOCIAL.RECIPROCITY_ENABLED` is true, each gift is recorded as a debt from
the receiver to the giver. After `REPAY_DELAY` iterations the debtor attempts a
repayment with its own hidden `repayment_probability`, returning
`REPAY_FRACTION` of the outstanding debt while staying at or above
`RESOURCE_FLOOR`. `REPAY_FRACTION` is constrained to `[0, 1]`; a value above one
would repay more than the debt. `repayment_probability` is drawn once at
initialization from `SOCIAL.REPAY_PROBABILITY_RANGE`; the legacy scalar
`REPAY_PROBABILITY` is still accepted for old configurations and sets every agent
to the same value. Repayments are ordinary money transfers, so they conserve
total money.

Unpaid debt does not vanish. A missed or partial attempt is a default: the
remainder is multiplied by `1 + DEFAULT_PENALTY_RATE`, retried after
`REPAY_RETRY_DELAY` iterations, and counted in `defaults` and
`total_defaults`. If either party dies, the unrecoverable
remainder is written off and recorded in `written_off_debt`. A creditor's death
is not observed as a borrower default and does not lower the borrower's
reputation. While a debtor has
an unpaid default with a creditor, `EXCLUDE_DEFAULTERS` prevents that creditor
from choosing the debtor as a new recipient until the debt is settled.

`repayment_log` records the debtor, creditor, original gift iteration,
repayment iteration, delay, amount, remaining balance, and whether the debt was
fully cleared. Metrics include `gift_volume`, `altruism_volume`,
`net_gift_volume`, `repayments`, `repayment_volume`, `reciprocity_rate`,
`repayment_event_rate`, `outstanding_debt`, `defaulted_debt`, `defaults`,
`total_defaults`, `written_off_debt`, and `mean_repayment_delay`.
`reciprocity_rate` is value-based (`repaid value / gifted value`, capped at 1);
`repayment_event_rate` is the old event-based count. `gift_volume` is gross
gifts, `altruism_volume` is gross gifts when reciprocity is off and net gifts
minus repayments when it is on, and `net_gift_volume` is the cumulative
difference.

Recipient memory is also a reciprocity channel. When explicit reciprocity is
enabled, `MEMORY.USE_WITH_RECIPROCITY: false` (the default) turns off the memory
preference so the same gift is not rewarded twice. Set it to true only when both
channels are wanted deliberately.

## Trust and reputation (optional)

Reputation is a belief about repayment, not gratitude for receiving a gift.
Disabled by default. When `TRUST.ENABLED` is true:

- every agent starts with public reputation and private beliefs equal to
  `INITIAL_BELIEF`;
- when a scheduled debt comes due, the creditor observes the fraction actually
  repaid (0 for a default, 1 for full repayment, or a partial value);
- the private belief of that creditor moves toward the observation by
  `LEARNING_RATE`, then blends with public reputation by `PUBLIC_WEIGHT`;
- public reputation moves toward the observed value (the outcome plus `BIAS`
  and Gaussian `NOISE`, clipped to [0, 1]);
- observations reach public reputation after `DELAY` iterations;
- beliefs and reputation are pulled back toward the prior by `DECAY` each
  iteration.

Gifts by themselves do not raise reputation. An agent that never repays is
therefore distinguishable from one that does, and with heterogeneous
`repayment_probability` reputation is statistically predictive of future
repayment.

Trust only records and scores behaviour; it does not move money or goods, so the
conservation ledger is unchanged. `reputation_error` compares public reputation
with each agent's observed repayment rate, while `reputation_brier` scores this
iteration's predictions against the observed repayment outcomes.

### Belief-based decisions

Reputation affects *receiver selection*, not the give or accept probability.
Counting the same belief in selection, giving, and acceptance would cube its
effect; a single channel keeps it linear and interpretable.

When `TRUST.DECISION_WEIGHT` is greater than 0 and reciprocity is enabled:

- risk-adjusted belief = `prior + risk_tolerance * (belief - prior)`;
- expected return per unit = `risk-adjusted belief * REPAY_FRACTION`;
- expected net payoff = `expected return - 1`;
- trust probability = logistic(
  `expected net payoff / DECISION_TEMPERATURE`);
- receiver selection multiplies each candidate's memory weight by
  `(1 - DECISION_WEIGHT) + DECISION_WEIGHT * trust_probability`.

Risk tolerance is not decision noise. A risk-tolerant agent acts on a strong
belief; a risk-averse agent falls back toward the prior. `DECISION_TEMPERATURE`
is a separate, fixed noise parameter, so raising risk tolerance does not make
every choice more random. If reciprocity is disabled, expected return is zero,
so trust cannot make giving look profitable.

Giving and acceptance use the legacy affordability
scores. Acceptance is not treated as free when reciprocity is enabled: existing
debt reduces the acceptance probability, because accepting creates a repayment
obligation with late fees, lower reputation, and loss of future credit.

`RISK_TOLERANCE` is drawn once per agent from `AGENTS.RISK_TOLERANCE` and stays
fixed (except for the optional shock adaptation described below). Each attempted
transfer is recorded in `decision_log` with the belief, risk tolerance,
probabilities, offered amount, and outcome, so decisions remain explainable.
Metrics report the mean give/accept probabilities.

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

While an active **scarcity** shock is running and the market is enabled, living
agents adapt at the end of each iteration: unmet demand raises `risk_tolerance`
by `ADAPTATION_RATE` (capped at 1), while met demand lowers it by half that rate
(floored at 0). An empty schedule, an information-only shock, or a disabled
market no longer changes risk tolerance. Metrics report `shock_active`,
`shock_count`, `production_modifier`, `reputation_noise`,
`mean_risk_tolerance` over living agents, `iterations_since_shock`, and `alive`.
`recovery_time()` returns the number of iterations from the last **scarcity**
shock until unmet demand returns to its pre-shock level; it returns `None` for
information shocks because they do not move demand.

## Invariants

1. A social transfer, repayment, market trade, subsistence payment, or dividend
   never changes `total_money = total + commons_fund`.
2. Total goods `goods_total_all = goods_total + commons_goods` change only by
   `produced + regenerated - consumed - commons_waste`.
3. Subsistence payments are capped by `resources - floor`; the amount moved is
   `living_cost_paid`, and the nominal charge is `living_cost_due`.
4. No agent's money, the commons stock, or the commons fund becomes negative.
   `LIVING_COST`, `RESOURCE_FLOOR`, `INITIAL_RESOURCES`, giving/receiving
   floors, ranges, `MEMORY.SIZE`, and `AGENTS` are validated at construction, so
   a negative cost cannot create a negative fund.
5. Dead agents do not participate in any step.
6. Trust beliefs and public reputation always stay within [0, 1].
7. The trust layer never changes money or goods totals.
8. Market prices stay within `[MIN_PRICE, MAX_PRICE]`.
9. Outstanding debt and written-off debt are never negative, and an
   outstanding debt is settled or written off, never silently dropped.
10. Extracted usable goods plus extraction waste equal the raw commons goods
    removed.

## Open decisions

- Use the commons fund to pay for commons regeneration or public works instead
  of holding it as a sovereign-fund balance.
- Pass stranded goods to other agents when an owner dies.
- Decide whether production should respond to the market price.
- Decide whether private beliefs should fully replace the legacy memory deque.
