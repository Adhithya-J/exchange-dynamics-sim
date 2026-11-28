# Subsistence World: Rules and Conservation

This document is the source of truth for the goods-only world in
`worlds/subsistence/`. Update it whenever a rule changes, and keep the
conservation tests in `tests/subsistence/test_conservation.py` in sync.

## Units and state

| Unit | Stored on | Meaning |
| --- | --- | --- |
| Health | `Agent.health` | Whole-day food buffer. Zero is death. |
| Berries | `Agent.berries` | Whole food units. Perishable outside storage. |
| Wood | `Agent.wood` | Whole durable blocks used for storage and barter. |
| Crates | `Agent.crates` | Whole perishable containers; each has a durability. |
| Trust | `Agent.trust` | Public reputation in `[0, 1]` used by credit. |
| Obligation | `ObligationBook` | A whole-good claim with a creditor, debtor, and optional due date. |
| Berry commons | `SubsistenceWorld.commons_berries` | Shared, regenerating whole-berry stock. |
| Wood commons | `SubsistenceWorld.commons_wood` | Shared, regenerating whole-wood stock. |
| Price | `Agent.price_belief`, `last_market_price` | Fractional berries per whole unit of wood. |

Berries and wood are **always whole units**. A day produces whole berries and
whole blocks of wood; there are no fractional resource stocks, trades,
spoilage, gifts, or loan principals. Prices, price beliefs, time, labour, the
production frontier, and interest rates are continuous because they are prices
or plans rather than goods. A trade's berry leg is rounded to a whole number,
so a fractional price leaves a small indivisibility residual.

The price is not money: it is only a fractional berries-per-wood exchange
ratio. There is no stored unit of account and no medium of exchange. A loan is a
claim on future goods, not money.

## Feature layers

Optional behaviour lives in `worlds/subsistence/features/`:

| Layer | Flag | Depends on | Role |
| --- | --- | --- | --- |
| `TradeLayer` | `TRADE` | — | Spot barter in the market square. |
| `GiftingLayer` | `GIFTING` | — | Need-based one-way berry gifts. |
| `TrustLayer` | `TRUST` | — | Public credit reputation and its updates. |
| `ReciprocityLayer` | `RECIPROCITY` | `GIFTING`, `TRUST` | Turns gifts into soft zero-interest obligations. |
| `CreditLayer` | `CREDIT` | `TRUST` | Reciprocal or risk-priced loans of wood, berries, or crates. |
| `ShockLayer` | `SHOCKS` | — | Temporary commons-regeneration droughts. |

`STORAGE` and `SPOILAGE` are passive core layers: their rules are part of the
resource engine, but they still register as flags so dependencies and metrics
are uniform. Dependencies are resolved transitively at startup, so enabling
`RECIPROCITY` also enables `GIFTING` and `TRUST`, and enabling `CREDIT` also
enables `TRUST`. Unknown flags are rejected rather than silently ignored.

The layers share two services:

- `GoodsLedger` — the only path for moving whole berries, wood, and crates
  between agents; used by gifts, trade, loans, repayment, and collateral.
- `ObligationBook` — the only ledger for loans and soft gift obligations;
  used by `CreditLayer` and `ReciprocityLayer`.

This is why reciprocal gifting and reciprocal credit do not duplicate one
another: a gift is a transfer plus, when reciprocity is on, a soft obligation;
a loan is a transfer plus a formal obligation with a due date.

## Agent state

Each agent has:

- `berry_skill` and `wood_skill` — efficiency multipliers on the time costs;
- `specialisation_z` — the shared latent `z` that generated the skills;
- `time_per_iteration` — the finite working day, in minutes;
- `berry_time_cost` and `wood_time_cost` — base minutes per whole unit;
- `health` and `max_health`;
- `berries`, `wood`, and `crates`;
- `price_belief` — fractional berries per wood, used for planning and barter;
- `reserve_days_preference` — this villager's stable taste for a food reserve;
- `trust`, `defaults`, `repayments`, `loans_taken`, and `loans_given` —
  credit reputation and history;
- `wood_progress` — fractional chopping work carried toward the next block;
- `craft_progress` — minutes invested in the next crate;
- `storage_wood_per_container` and `storage_berries_per_container`;
- per-iteration observations: `labor_berries`, `labor_wood`, `labor_craft`,
  `planned_berries`, `planned_wood`, `planned_crates`, `crafted_crates`,
  `gathered_berries`, `gathered_wood`, `meals_eaten`, `unmet_food`,
  `post_trade_berries`, and `post_trade_wood`.

Agents start with ten berries, no raw wood, one finished crate, and full
health. They start with no money because money does not exist in this world.

Skills are drawn from one shared specialisation latent variable `z` per
villager, so they are **negatively correlated** by construction. A high `z` is
a berry specialist; a low `z` is a wood specialist. The default configuration
draws `z` uniformly from `[-2, 2]` and then clips:

```text
berry_skill = clip(0.85 + 0.35 * z, 0.30, 1.50)
wood_skill  = clip(1.05 - 0.25 * z, 0.45, 1.45)
```

With `BERRY_TIME_COST = 10`, `WOOD_TIME_COST = 120`, and a 100-minute day,
this gives roughly 3–15 berries/day and 0.4–1.2 wood/day. The wood spread is
deliberately narrower so top choppers do not strip the forest immediately.

## A finite day and the production possibilities frontier

One iteration is a finite day of `TIME_PER_ITERATION` minutes. Picking one
berry costs:

```text
BERRY_TIME_COST / berry_skill    minutes
```

Chopping one block of wood costs:

```text
WOOD_TIME_COST / wood_skill      minutes
```

Chopping is deliberately much slower than picking, and both activities become
slower as the commons thins out. The model tracks each stock's density as its
current stock divided by its initial stock, capped at 1:

```text
berry_density = min(1, commons_berries / initial_commons_berries)
wood_density  = min(1, commons_wood  / initial_commons_wood)
```

Sparse commons add search and travel time. With a floor of `MIN_DENSITY`
(default `0.10`), the effective minutes per unit are:

```text
effective_berry_minutes =
    BERRY_TIME_COST / (berry_skill * max(berry_density, MIN_DENSITY))
effective_wood_minutes =
    WOOD_TIME_COST / (wood_skill * max(wood_density, MIN_DENSITY))
```

The floor keeps costs finite even when a stock is empty. The maximum daily
yields therefore contract with density:

```text
max_berries = TIME_PER_ITERATION / effective_berry_minutes
max_wood    = TIME_PER_ITERATION / effective_wood_minutes
```

At full density these reduce to the familiar skill-only yields. The PPF chart
uses these density-adjusted yields for the selected iteration, so the frontier
visibly contracts as bushes or trees become sparse.

With all available time used, the frontier is the straight line:

```text
berries / max_berries + wood / max_wood = 1
```

The opportunity cost of one block of wood is:

```text
max_berries / max_wood    berries per wood
```

A villager with a low opportunity cost has a comparative advantage in wood; a
villager with a high opportunity cost has a comparative advantage in berries.
The PPF chart plots this line, the planned labour point, the actual gathered
point, and the holding after barter.

The daily yield is planned in fractional units but harvested as whole units.
Berries use stochastic rounding: a planned yield of `2.4` berries becomes two
berries, with a 40% chance of a third, using a seeded random draw. This keeps
the expected output equal to the time spent while keeping every berry stock
whole.

Wood uses a carried fractional progress instead. Chopping is slow, so a
villager may not finish a whole block in one day. `wood_progress` holds the
fraction of the next block already completed; each day adds that day's planned
wood, extracts every whole block now available, and keeps the remainder for
tomorrow. Wood stocks therefore stay whole without throwing away slow work.

### Labour allocation

With `FEATURES.TRADE: true`, an agent first samples the activity with the
higher density-adjusted market value per minute. Each value carries a daily
log-normal taste shock, and with probability `EXPLORATION_RATE` the two values
are swapped:

```text
berry value per minute =
    berry_skill * max(berry_density, MIN_DENSITY) / BERRY_TIME_COST
wood value per minute =
    wood_skill * max(wood_density, MIN_DENSITY) * price / WOOD_TIME_COST
```

- A **wood specialist** picks up to its daily meal if the larder is short, then
  finishes or advances a crate when raw wood is available, then chops. It does
  not gather a large berry reserve of its own; it sells wood for food.
- A **berry specialist** covers today's meal, then refills the post-meal
  reserve only when it has fallen below the refill threshold. Above the
  threshold it leaves the larder alone and uses the day for carpentry or
  chopping.
- Berry pickup is capped at one daily ration plus the empty space left in the
  crates: `FOOD_PER_ITERATION + max(0, storage_capacity - berries)`. A refill
  batch therefore never picks berries that must spoil, and leftover minutes go
  to wood or carpentry instead.
- A wood worker keeps `WOOD_RESERVE_CRATES` spare crate's worth of raw wood
  once it owns a crate, so it can replace a worn crate without a fresh chopping
  binge.
- Neither specialist has to produce everything itself; the market is how the
  two sides cover their deficits.

With `FEATURES.TRADE: false`, agents follow a self-sufficiency rule: today's
food, then wood for storage, then a berry reserve, then extra berries only if
spoilage is off.

## Shared commons

The world has two independent commons, `BERRIES` and `WOOD`. Each has whole
`INITIAL` and `CAPACITY` values, a fractional `REGENERATION` rate, and an
`EXTRACTION_EFFICIENCY` in `(0, 1]`.

Each iteration the stock grows by a whole number close to the regeneration
rate. A fractional rate such as `0.25` means a one-block chance of regrowth:

```text
whole = floor(REGENERATION)
fraction = REGENERATION - whole
regenerated = whole + (1 with probability fraction)
regenerated = min(regenerated, CAPACITY - stock)
```

An agent requests a whole number of units, obtained by stochastic rounding of
its planned yield. The usable amount is limited by the stock and efficiency:

```text
usable_cap = floor(stock * EXTRACTION_EFFICIENCY)
usable = min(requested, usable_cap)
raw = usable if efficiency == 1 else ceil(usable / efficiency)
stock -= raw
agent.stock += usable
waste = raw - usable
```

Chopping is slow, so a fractional wood yield is carried instead of rounded.
The agent keeps the fraction of the next block (always below one block) and only
whole blocks enter storage. A villager whose effective yield is 0.4 wood per day
therefore produces one block every third day, never a surprise block. When the
forest cannot supply the planned whole blocks, the shortfall is recorded as
`wood_blocked`: a partially chopped block is kept, while unfulfilled whole
blocks are wasted, like standing at an empty copse. This keeps the daily plan
auditable against the gathered total.

Berries regrow quickly and are abundant. Wood regrows much more slowly and is
the scarce, valuable resource. Density is measured against the initial stock
and capped at 1, so a stock above its initial level is still treated as full;
the `MIN_DENSITY` floor of `0.10` prevents division by zero when a stock is
exhausted. Each iteration records `berries_commons_pct` and
`wood_commons_pct` for the harvesting phase, and the viewer uses those values
for both the frontier and the multi-site search animation.

### Village size

By default the commons scales with the village: `SCALE_WITH_AGENTS` multiplies
`INITIAL`, `CAPACITY`, and `REGENERATION` by `AGENTS / REFERENCE_AGENTS`
(`REFERENCE_AGENTS` defaults to 10). A 50-agent village therefore has five times
the berry and wood flow of the baseline, so the same per-villager ecology holds
at any size. The scaling is recorded in `meta.commons`; set it to `false` to
use the raw `COMMONS` values at every village size.

### Seasons

`SEASONS` splits time into a repeating cycle (spring, summer, autumn, winter by
default), `LENGTH` iterations each. Each season multiplies regeneration:

```text
regenerated = stochastic_round(REGENERATION * shock_factor * season_factor)
```

The default berry factors are `1.00`, `1.40`, `0.80`, and `0.00`; wood stays at
`1.00` all year. Winter therefore stops berry regrowth entirely while wood keeps
coming in. The commons is the village granary: a summer that strips the bushes
leaves nothing to draw on when the berries stop, so saving berries (or leaving
stock standing) is the only way through the winter. Season state is recorded per
iteration as `season` (cycle index), `season_name`, `berry_regen_factor`, and
`wood_regen_factor`, and the viewer tints the scene and labels the season.

## Health and eating

Every living agent needs a full daily ration of `FOOD_PER_ITERATION` whole
berries. Health is a **whole-day buffer**, not a count of individual berries.

```text
meal = min(agent.berries, FOOD_PER_ITERATION)
agent.berries -= meal
agent.unmet_food = FOOD_PER_ITERATION - meal
```

If the ration is full:

```text
health = min(MAX_HEALTH, health + HEAL_PER_MEAL)
```

Otherwise the day is short:

```text
health -= STARVATION_RATE
```

Death is `health <= 0`. With the defaults (`MAX_HEALTH = 4`,
`STARTING_HEALTH = 4`, `FOOD_PER_ITERATION = 6`, `HEAL_PER_MEAL = 1`,
`STARVATION_RATE = 1`), a full-health agent survives exactly four days without
food, and a full day's ration restores one point of health up to the maximum. A
short day costs one point regardless of how many berries were missing. The
four-day buffer is deliberate: missing the six-berry ration creates immediate
early-game scarcity and a reason to barter instead of a ten-day cushion.

Dead agents do not gather, gift, trade, or eat. Their goods remain in their
inventory and are still counted in world totals; their crates continue to wear
out and their unprotected berries continue to spoil.

## Storage, crates, and wear

Raw wood is material, not storage. When `FEATURES.STORAGE` is true:

```text
capacity = crate_count * BERRIES_PER_CONTAINER
```

With the defaults, one finished crate stores ten berries. A villager starts with
one crate and ten berries, enough to save food and material while working toward
the next crate. If every crate wears out, storage capacity is zero and the
villager can only carry today's food until a new crate is finished.

### Food reserve and refill threshold

A villager keeps a **post-meal** reserve of `RESERVE_DAYS` days of food,
subject to capacity:

```text
reserve_target = min(capacity, RESERVE_DAYS * FOOD_PER_ITERATION)
```

With two days and a six-berry ration the desired reserve is twelve berries. One
crate can hold only ten, so the effective target is ten; two crates hold twelve
of their twenty slots, and the second crate is useful without being filled to
the brim. A small
per-villager `RESERVE_JITTER` varies the target, so stocks differ across the
population.

Villagers do not top up a few berries every single day. They refill in a batch
only when the larder falls below `REFILL_FRACTION` of the reserve target, and
otherwise spend whole days on wood or carpentry. A below-target wood worker
first picks up to its daily meal if it can, then specialises.

Raw wood is also held as a small replacement reserve. Once a villager owns a
crate, it chops toward `WOOD_RESERVE_CRATES` spare crate's worth of material in
addition to any missing crate. Raw wood is durable, so this is the cheapest way
to be ready when a crate wears out.

### Skilled crate crafting

Crafting one crate requires `WOOD_PER_CONTAINER` raw wood and:

```text
craft_minutes = BASE_CRAFT_MINUTES / wood_skill
```

`BASE_CRAFT_MINUTES` is 100, so a wood worker at skill 1 builds a crate in one
day, a top chopper does it in about 70 minutes, and a weak carpenter takes two
or three days. `craft_progress` carries partial minutes between iterations. On
completion, the raw wood is consumed, a crate with durability 1.0 is added, and
the remaining craft minutes carry forward.

Crates are perishable. Every iteration a crate loses:

```text
wear = CRATE_WEAR_RATE * (1 + CRATE_USE_WEAR_FACTOR * utilization)
```

where utilization is stored berries divided by total capacity. A busy crate
wears out faster than an idle one. A crate whose durability reaches zero is
lost.

When `FEATURES.SPOILAGE` is true, berries above capacity spoil:

```text
spoiled = max(0, berries - capacity)
berries -= spoiled
```

If `STORED_SPOIL_RATE` is positive, stored berries also lose a whole number
chosen by stochastic rounding. With `FEATURES.SPOILAGE: false`, storage has no
preservation value. With `FEATURES.STORAGE: false`, every uneaten berry spoils.

## Decentralized price discovery and barter

Barter is optional (`FEATURES.TRADE`). There is no global price and no money.
Each villager has a fractional `price_belief` in berries per wood, initialized
near the village baseline opportunity cost with a small random offset. There is
no MIN_PRICE, MAX_PRICE, belief band, or crate-price ceiling: the market is
allowed to discover any positive exchange ratio. Only positivity is enforced so
that a ratio remains meaningful.

### Planning with beliefs

A villager plans the day using its own `price_belief` and the latest observed
commons density. The wood-versus-berries choice uses:

```text
berry value/minute = berry_skill * density / BERRY_TIME_COST
wood value/minute  = wood_skill * density * price_belief / WOOD_TIME_COST
```

Each day the villager draws a log-normal taste shock on each value and, with
probability `EXPLORATION_RATE`, swaps the two activities when both activities
have positive value. This is deliberate sampling rather than a deterministic
argmax: choices vary slightly around the rational comparison, and a villager
occasionally tries the other activity. An activity with zero value is not
explored, because that would make an agent do worthless work.

Carpentry is a third use of the day. A wood specialist first covers its meal if
it can, then finishes or advances a crate when raw wood is available, then
chops or gathers with the remainder. Berry specialists cover food and the crate
reserve before using spare minutes for carpentry or chopping. A non-urgent craft
is occasionally deferred for a day of chopping.

### Reservation prices

At the start of the barter phase every living villager's start-of-tick belief
is snapshotted. Reservations are computed from that snapshot, the villager's
effective opportunity cost, and survival pressure:

```text
wood seller min    = blend(opportunity, belief) or subsistence floor
wood buyer max     = blend(opportunity, belief) + storage urgency
crate seller min   = blend(crate opportunity, crate belief) * durability
crate buyer max    = blend(crate opportunity, crate belief) + urgency
```

The blend is mostly belief with a smaller opportunity-cost anchor. A wood
buyer may spend saved berries down to today's meal, so savings can smooth a thin
market instead of limiting bids to the day's production surplus. A seller who is
short of food *lowers* its ask; a near-starving seller accepts almost any
positive price, because eating today matters more than the quoted ratio. When a
seller has enough food, the subsistence floor is the berry cost of a day's meal
from one day's wood output. Crate opportunities include the eight raw wood and
the craft minutes valued at the villager's berry rate, scaled by the durability
of the crates actually offered.

### Bilateral matching

The market square runs two order books, crates first and then raw wood:

- sellers have a private minimum acceptable price and a whole-unit quantity;
- buyers have a private maximum willingness to pay, a quantity, and a berry
  budget;
- the cheapest asks are matched against the highest bids first, with a random
  tie-break, so a desperate seller reaches the buyers who can actually pay;
- an agent cannot trade with itself and cannot be both a buyer and seller of the
  same good in one iteration.

If `seller_min <= buyer_max`, the trade clears at the fractional midpoint:

```text
trade_price = (seller_min + buyer_max) / 2
```

Prices are fractional. Every trade moves whole goods and conserves them. For a
wood trade the agreed price is berries per wood; for a crate trade it is berries
per crate. The berry leg is rounded to the nearest whole berry (at least one)
and the whole quantity is reduced by one unit if rounding would breach the
buyer's budget. That residual is an explicit indivisibility cost, not a price
clamp.

Wood and crate prices are kept separate. A crate price bundles wood, craft
time, and storage, so it does not update the wood price belief or the headline
berries-per-wood price. The market price in the header is the realized average
wood price for the iteration, or the last observed wood price when no wood trade
occurred.

### Information lag and gossip

All reservations and trades use the start-of-tick beliefs. Belief updates are
batched until the barter phase ends, so they take effect on tick `t + 1`:

- wood traders move one berry toward their realized wood trade price;
- other villagers who actually entered the market square move one gossip step
  toward the average wood trade price;
- a fully unmatched wood seller lowers its belief by one unless the ask is
  already at its subsistence floor; a fully unmatched wood buyer raises it by one
  unless the bid is already capped by its berry budget.

Beliefs are fractional and have no band or upper bound; only positivity is
enforced. Villagers who did not enter the market do not receive gossip and do
not change their beliefs.

## Gifting and reciprocity

Gifting is off by default (`FEATURES.GIFTING: false`). When enabled,
`utils.social.process_berry_gifts` lets a living agent with whole berries above
`MIN_RESERVE` give a whole amount from `AMOUNT_RANGE` to the neediest living
agent, chosen by lowest health and then lowest berries. The move goes through
`GoodsLedger`, so gifts conserve the berry total.

With `FEATURES.RECIPROCITY: true`, each gift also creates a **soft obligation**
in the shared `ObligationBook`: zero interest, no hard due date, and the same
whole-berry amount. The receiver repays when it has berries above its daily meal
plus `RECIPROCITY.REPAY_THRESHOLD`, up to `MAX_OUTSTANDING` outstanding berries.
Gifts that would push a receiver past `MAX_OUTSTANDING` are capped before they
move, so the cap never turns a reciprocal gift into a free transfer. A soft gift
that is never serviced after `RECIPROCITY.DEFAULT_AFTER_DAYS` is written off,
lowers the debtor's trust, and frees the cap. If either side dies the claim is
written off without penalising the borrower.

This is deliberately the low-commitment end of the credit spectrum. A plain gift
is a one-way transfer; a reciprocal gift is a transfer plus a soft claim; a
formal loan is a transfer plus a dated claim.

## Credit

Credit is off by default (`FEATURES.CREDIT: false`) and requires `TRUST`, which
is auto-enabled. Both modes move whole **wood**, **berries**, or **crate
capacity** now and require the same good later.

### Borrowers and lenders

A borrower has demand when it is short of food, lacks a crate, or lacks the wood
for the next crate. Food is the first claim: a starving villager borrows today's
meal before crate material or wood. A lender supplies wood above
`LENDER_WOOD_BUFFER`, berries above its meal plus `LENDER_BERRY_BUFFER`, or
crate capacity above `LENDER_CRATE_RESERVE` and above any crates already pledged
as collateral. Pledged crates are excluded from both the crate market and new
lending, so a borrower cannot sell or lend the security for an existing loan. A
borrower may owe at most `REPAYMENT_CAPACITY_FRACTION` of its
expected net production over `TERM_DAYS`, and each side has a concurrent-loan
limit. For berries, expected production includes wood sold for berries and is net
of the borrower's own food need. For crates, capacity requires both the wood and
the craft time to make a crate.

Crate lending is direct capacity lending: the lender transfers one crate, the
borrower uses the storage, and repayment is one crate. The lent crate is also
the collateral, so a default can return it to the lender if it still exists.
Crate durability is tracked: a new crate covers a loan fully, a worn crate covers
only its remaining durability, and market mode prices the residual risk. In
reciprocal mode crate loans remain zero-interest.

### Reciprocal mode

`MODE: reciprocal` is zero interest: the borrower owes exactly the principal.
A loan is allowed when the borrower's `trust` is at least `MIN_TRUST`, or when
it pledges a crate. A low-health borrower *without* collateral is refused; a
low-health borrower *with* collateral is not. This is the formal, dated version
of the trust-and-reciprocity channel.

### Market mode

`MODE: market` prices default risk. A borrower below `MIN_MARKET_TRUST` must
pledge a crate; a secured low-trust borrower is allowed. A borrower's repayment
probability is:

```text
p = clip(0.25
         + 0.35 * health / max_health
         + 0.20 * min(1, berries / food)
         + 0.20 * trust,
         0.15, 0.98)
```

The break-even default premium is `(1 - p) / p`. The quoted interest rate is:

```text
rate = clamp(BASE_INTEREST + RISK_PREMIUM * (1 - p) / p,
             MIN_INTEREST, MAX_INTEREST)
```

and a pledged crate reduces it by up to `COLLATERAL_DISCOUNT` times the
durability-adjusted collateral coverage. A fully covering crate makes the loan
risk-free (zero quoted interest); a worn crate covers only its remaining
durability. Repayment is:

```text
repayment = max(principal, ceil(principal * (1 + rate)))
```

The realized interest is `repayment / principal - 1`, shown in the replay.
Because repayment moves whole goods, a small positive quoted rate can round up
to one extra whole unit; that is the accepted indivisibility cost, not a second
price clamp. There is no money: this is a **real** goods interest rate.

### Repayment, grace, and default

At the start of each tick, due formal loans are settled **before** soft gift
obligations. A berry debtor keeps `FOOD_PER_ITERATION` berries before servicing
a formal loan, so debt cannot consume today's meal. A borrower repays what it can
above that reserve; any remainder enters grace or default. A borrower that is
short gets `GRACE_DAYS` extra days. After that, or on death, the obligation
defaults:

1. pledged crates are seized, highest-durability first;
2. the remaining principal is taken from the borrower's stock of the loan good,
   again leaving a living berry debtor's daily meal untouched;
3. the rest is a loss for the lender.

A default lowers the borrower's trust by `TRUST.LOSS`; a repayment raises it by
`TRUST.GAIN`, scaled by the fraction of principal repaid. If the creditor dies
first, the claim is written off without penalising the borrower. Goods are never
destroyed by default: they either move to the creditor or stay with the debtor.
If a seized crate is worth more than the remaining claim, the excess is recorded
as `collateral_excess`; it is an explicit whole-crate indivisibility, not hidden
as zero loss.

## Shocks

`FEATURES.SHOCKS` adds a temporary drought. Each tick without an active shock
starts one with probability `SHOCKS.PROBABILITY`, lasting a whole number of days
from `DURATION_RANGE`. While active, berry and wood regeneration are multiplied
by `BERRY_REGEN_FACTOR` and `WOOD_REGEN_FACTOR`. Shocks are logged and shown in
the replay; the conservation ledger uses the actual regenerated amount.
Re-initializing a world clears any active drought, so a fresh run never starts
mid-shock.

## Death

Death is explicit: `health <= 0`. A death is logged once with cause `hunger`.
Dead agents are skipped by every living-agent step. Their goods remain counted
until they wear out or spoil, so a death never silently deletes resources.

## Iteration order

1. Reset per-iteration observations.
2. Optional shock roll; set berry and wood regeneration factors.
3. Regenerate the commons as whole units, then record
   `berries_commons_pct` and `wood_commons_pct` for the harvest.
4. Settle due formal loans first (preserving a living berry debtor's daily
   meal), then repay soft gift obligations when possible.
5. Allocate the finite day using density-adjusted costs and gather; living
   agents are processed in random order, berries before wood within an agent.
6. Optional need-based whole-berry gifts; with reciprocity, the gift also
   creates a soft obligation.
7. Optional bilateral barter: crates first, then raw wood, using start-of-tick
   beliefs; batch all belief and gossip updates for tick `t + 1`.
8. Optional credit origination.
9. Record each living agent's post-trade holdings.
10. Eat and update health; log any deaths.
11. Apply crate wear and berry spoilage to every agent, including the dead.
12. Optional trust decay.
13. Clamp stocks to non-negative whole values and health to
    `[0, MAX_HEALTH]`.
14. Record metrics and an agent snapshot.

## Conservation ledger

Let `B = sum(agent.berries) + commons_berries` and
`W = sum(agent.wood) + commons_wood`.

| Process | Berries | Wood | Notes |
| --- | --- | --- | --- |
| Commons regeneration | **source** | **source** | Adds whole units up to `REGENERATION`, capped by `CAPACITY`. |
| Gathering | moves + **sink** | moves + **sink** | Usable goods move from commons to the agent; extraction waste is a sink. |
| Gift | conserved | unchanged | The giver loses exactly what the receiver gains. |
| Barter | conserved | conserved | Whole berries move against whole wood or finished crates at agreed prices. |
| Loan origination | conserved | conserved | The lender loses exactly the principal the borrower gains; crates move whole. |
| Loan repayment | conserved | conserved | The borrower returns exactly the agreed amount of the loan good. |
| Default | conserved | conserved | Collateral and estate goods move to the lender; a shortfall is a claim loss, not a goods sink. |
| Carpentry | unchanged | **sink** | Consumes `WOOD_PER_CONTAINER` raw wood and adds one finished crate. |
| Eating | **sink** | unchanged | Living agents consume up to `FOOD_PER_ITERATION` whole berries. |
| Spoilage | **sink** | unchanged | Whole berries above storage capacity, plus optional stored spoilage. |
| Raw wood wear | unchanged | **sink** | Whole blocks lost to wear, including on dead agents. |
| Crate wear | unchanged | unchanged | Crates lose durability and can disappear as a separate storage stock. |
| Death | unchanged | unchanged | Goods remain in the dead agent's inventory. |

The exact invariants are:

```text
ΔB = regenerated_berries - eaten - spoiled - commons_waste_berries
ΔW = regenerated_wood - wood_decayed - wood_crafted - commons_waste_wood
```

`tests/subsistence/test_conservation.py` enforces these every iteration,
including runs with gifting, trade, credit, reciprocity, shocks, extraction
waste, and deaths.

## Feature flags

Flags live under `FEATURES`:

| Flag | Default | Effect |
| --- | --- | --- |
| `GIFTING` | `false` | Need-based whole-berry gifts from the shared social utility. |
| `TRADE` | `true` | Bilateral whole-unit barter in wood and finished crates at decentralized prices. |
| `CREDIT` | `false` | Deferred-payment loans of wood, berries, or crate capacity; mode is `CREDIT.MODE`. |
| `STORAGE` | `true` | Finished crates provide storage capacity. |
| `SPOILAGE` | `true` | Unprotected berries spoil at the end of the iteration. |
| `TRUST` | `false` | Public credit reputation. Auto-enabled by `CREDIT` and `RECIPROCITY`. |
| `RECIPROCITY` | `false` | Gifts create soft zero-interest obligations. Auto-enables `GIFTING` and `TRUST`. |
| `SHOCKS` | `false` | Temporary droughts reduce commons regeneration. |

Dependencies are resolved transitively. The flags are otherwise independent. For
example:

- `STORAGE: false`, `SPOILAGE: true` — a harsh world where only today's food
  can be carried.
- `SPOILAGE: false` — berries keep indefinitely, so wood has no preservation
  value and barter demand for it disappears.
- `TRADE: false` — self-sufficiency; agents use the priority heuristic and
  surplus time is left idle after their targets are met.
- `GIFTING: true` — one-way berry transfers beside barter.
- `GIFTING: true`, `RECIPROCITY: true` — gifts become soft zero-interest
  obligations; this is the low-commitment credit channel.
- `CREDIT: true`, `CREDIT.MODE: reciprocal` — dated zero-interest loans gated by
  trust or collateral.
- `CREDIT: true`, `CREDIT.MODE: market` — the same loans with risk-priced
  goods interest.
- `SHOCKS: true` — occasional droughts cut commons regeneration.
- `SEASONS.ENABLED: false` — a steady climate; the commons regenerates at the
  same rate all year.
- `COMMONS.SCALE_WITH_AGENTS: false` — the raw commons values regardless of
  village size, so big villages are poor by construction.

## Replay schema (version 3)

`worlds/subsistence/replay.py` writes and validates schema version 3 with
`meta.world = "subsistence"`. Version 2 replays still load.

- `meta` — generator, world, iterations, seed, agent count, the ordered
  `agent_state_fields`, effective feature flags, health settings, finite-day
  time costs, crate crafting and wear settings, fractional-pricing marker,
  credit/trust/reciprocity/shock settings, season cycle, commons scaling and
  effective commons settings, resource definitions, initial commons and totals,
  the initial market price, and the full config.
- `agents` — id, label, colour, berry and wood skills, the `specialisation_z`
  latent, **initial** price belief, time budget, time costs, craft minutes, base
  maximum daily yields, **initial** health and crate count, and the storage
  container ratio.
- `ticks` — one row per iteration plus iteration zero. Each tick has summary
  metrics, including commons density, market price, beliefs, crate counts,
  credit originations/defaults, outstanding debt, mean trust, shock state, and
  season, and one state per agent. Version 3 stores each agent state as a
  positional row whose field order is `meta.agent_state_fields`: id, whole
  health, berries, raw wood, crates with durability, storage capacity, crafting
  progress, price belief, trust, debt and credit receivables, the berry/wood/
  craft labour split, density-adjusted effective yields, carried wood progress,
  planned goods, actual gathered goods, post-trade holdings, and alive status.
  Positional rows cut a large replay to roughly a quarter of its version 2 size
  and parse much faster at 50-100 agents.
- `events` — typed `gather`, `craft`, `gift`, `trade`, `loan`, `repay`,
  `default`, `write_off`, `late`, `shock`, `spoil`, and `death` records with a
  tick, phase, sequence number, and stable id. Trades carry the goods type, whole
  quantities, fractional price, and whole berry payment; credit events carry
  principal, repayment, interest, due date, durability-adjusted collateral,
  recovery, and collateral excess; shock events carry duration and regeneration
  factors.

The viewer reads this replay and never imports the model. `decode_tick_agents`
turns a tick's states into mappings for analysis code.

## Metrics

The world records, among others: `alive`, `mean_health`, `total_berries`,
`total_wood`, `total_crates`, `crate_durability`, `crates_crafted`,
`crates_worn`, `wood_crafted`, `commons_berries`, `commons_wood`,
`berries_commons_pct`, `wood_commons_pct`, `gini_berries`, `gini_wood`,
`barter_price`, `last_crate_price`, `mean_price_belief`, `market_participants`,
`market_traders`, `failed_trades`, `berries_gathered`, `wood_gathered`,
`wood_blocked`, `berries_eaten`, `unmet_food`, `berries_spoiled`, `wood_decayed`,
`commons_waste_berries`, `commons_waste_wood`, `storage_capacity`,
`storage_used`, `storage_utilization`, `trade_berries`, `trade_wood`,
`crate_trades`, `trade_crates`, `gifts`, `gift_berries`, `mean_trust`,
`loans_originated`, `loan_principal_wood`, `loan_principal_berries`,
`loan_principal_crates`, `mean_interest_rate`, `loans_repaid`,
`repayment_wood`, `repayment_berries`, `repayment_crates`, `loans_defaulted`,
`default_loss_wood`, `default_loss_berries`, `default_loss_crates`,
`loans_late`, `collateral_seized`, `collateral_excess_value`, `loans_active`,
`credit_outstanding_wood`, `credit_outstanding_berries`,
`credit_outstanding_crates`, `written_off_wood`, `written_off_berries`,
`written_off_crates`, `gift_obligations`, `gift_obligation_berries`,
`gift_repayments`, `gift_defaults`, `gift_default_berries`, `shock_active`,
`shocks`, `season`, `season_name`, `berry_regen_factor`, `wood_regen_factor`,
`specialisation`,
`ppf_efficiency`, and `extinction_iteration`.

Metrics are per iteration. The experiment runner aggregates flow metrics
(`loans_originated`, `trade_berries`, `gifts`, `berries_spoiled`, and so on) over
the run and leaves stock metrics at the final value; reporting only the last
tick would make most credit and trade experiments look empty.

`specialisation` is a normalised Herfindahl index over berry, wood, and craft
labour: 0 means the day is split evenly across all three, 1 means one activity
uses the whole day.

`ppf_efficiency` compares the berry value of actual gathering with the berry
value of the planned labour at the current fractional price. A value below 1 means
the commons limited the plan.

## Invariants

1. Agent berries, raw wood, crates, the commons stocks, and health never become
   negative, and berries, wood, crate counts, storage capacity, and health are
   whole numbers.
2. Health stays within `[0, MAX_HEALTH]`.
3. Labour spent on berries, wood, and crafting never exceeds the agent's finite
   day.
4. `wood_progress` and `craft_progress` are fractional and carry between
   iterations; craft progress is not counted as a finished crate.
5. `berries_commons_pct` and `wood_commons_pct` stay in `[0, 1]`, and effective
   minute costs use `max(density, MIN_DENSITY)` so they remain finite.
6. Total berries and total raw wood change only by the sources and sinks in the
   ledger above; carpentry consumes raw wood and creates a crate.
7. Goods transfers conserve both goods. Gifts, trades, loan origination,
   repayment, collateral, and estate seizure only move goods between agents;
   default losses are claim losses, not goods sinks.
8. Dead agents do not participate in gathering, crafting, gifting, trade,
   credit origination, or eating. Their estate can still be seized for a
   defaulted loan.
9. Every agent's price belief and every trade price is positive and fractional;
   there is no upper bound, band, or price clamp. Goods quantities and the berry
   leg of each trade are whole.
10. Storage capacity is `crate_count * BERRIES_PER_CONTAINER` while storage is
    enabled, and zero otherwise.
11. Crate durability stays in `(0, 1]`; crates at zero durability are removed.
    Crate prices, collateral values, and credit coverage scale with durability.
12. Trust stays in `[0, 1]`. Formal loan principals and repayments are whole
    units, repayments are at least the principal, and reciprocal-mode
    repayments equal the principal.
13. Soft gift obligations are zero-interest and have no hard due date; formal
    loans have a due iteration and a finite grace period. A soft obligation older
    than `RECIPROCITY.DEFAULT_AFTER_DAYS` becomes a trust-penalising write-off
    rather than an indefinite free claim.
14. A living berry debtor keeps `FOOD_PER_ITERATION` berries before formal loan
    service or default seizure. A borrower may repay partially above that reserve.
15. A creditor's death is a write-off, never a borrower default.
16. `SIMULATION.AGENTS` must be at least one.

## Open decisions

- Whether the shared commons should have spatial access or ownership rules.
- Whether wood should have a second use, such as tools or shelter.
- Whether stored berries should spoil slowly by default.
- Whether agents should learn their comparative advantage rather than read
  their current price belief directly.
- How crate durability should vary by material or craft skill.
- Whether over-collateral seizure should return change instead of recording a
  `collateral_excess` whole-crate transfer.
