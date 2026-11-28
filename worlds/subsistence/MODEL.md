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
| Berry commons | `SubsistenceWorld.commons_berries` | Shared, regenerating whole-berry stock. |
| Wood commons | `SubsistenceWorld.commons_wood` | Shared, regenerating whole-wood stock. |
| Price | `SubsistenceWorld.trade_price` | Whole berries given per whole unit of wood. |

Berries and wood are **always whole units**. A day produces whole berries and
whole blocks of wood; there are no fractional resource stocks, trades,
spoilage, or gifts. Time, labour, and the production frontier are continuous,
because they are plans rather than goods.

The price is not money: it is only a whole-berry-per-wood exchange ratio. There
is no stored unit of account and no medium of exchange.

## Agent state

Each agent has:

- `berry_skill` and `wood_skill` — efficiency multipliers on the time costs;
- `specialisation_z` — the shared latent `z` that generated the skills;
- `time_per_iteration` — the finite working day, in minutes;
- `berry_time_cost` and `wood_time_cost` — base minutes per whole unit;
- `health` and `max_health`;
- `berries` and `wood`;
- `wood_progress` — fractional chopping work carried toward the next block;
- `storage_wood_per_container` and `storage_berries_per_container`;
- per-iteration observations: `labor_berries`, `labor_wood`,
  `planned_berries`, `planned_wood`, `gathered_berries`, `gathered_wood`,
  `meals_eaten`, `unmet_food`, `post_trade_berries`, and
  `post_trade_wood`.

Agents start with the configured starting berries and wood (default zero) and
full health. They start with no money because money does not exist in this
world.

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

Chopping is deliberately much slower than picking. The maximum daily yields
are therefore:

```text
max_berries = TIME_PER_ITERATION * berry_skill / BERRY_TIME_COST
max_wood    = TIME_PER_ITERATION * wood_skill  / WOOD_TIME_COST
```

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

With `FEATURES.TRADE: true`, an agent first chooses the activity with the
higher market value per minute:

```text
berry value per minute = berry_skill / BERRY_TIME_COST
wood value per minute  = wood_skill * price / WOOD_TIME_COST
```

- A **wood specialist** spends enough time to finish the next whole block
  (using any carried `wood_progress`), then uses the rest of the day to gather
  food and its berry reserve. This lets a low-berry villager produce a tradable
  block even though a full ration would otherwise consume the whole day.
- A **berry specialist** covers today's food and its berry reserve, then
  produces a small surplus to buy one block of wood plus `TRADE_SURPLUS`.
- A berry specialist does not chop its own wood; it buys wood with berries.
- A wood specialist does not have to gather all of its own berries; it can sell
  wood for them.

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

Berries regrow quickly and are abundant. Wood regrows much more slowly and is
the scarce, valuable resource.

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
inventory and are still counted in world totals; their wood continues to wear
out and their unprotected berries continue to spoil.

## Storage, spoilage, and wood wear

When `FEATURES.STORAGE` is true, an agent's storage capacity is:

```text
capacity = (wood * BERRIES_PER_CONTAINER) // WOOD_PER_CONTAINER
```

The default is **eight wood blocks for ten berries**, so eight wood stores ten
berries, four wood stores five, and one wood stores one. Capacity is always a
whole number of berries.

When `FEATURES.STORAGE` is false, capacity is zero.

Wood wears out as whole blocks. Each block has a `WOOD_DECAY_CHANCE` per
iteration:

```text
lost = number of blocks whose decay draw succeeds
wood -= lost
```

When `FEATURES.SPOILAGE` is true, berries above the post-wear capacity spoil:

```text
spoiled = max(0, berries - capacity)
berries -= spoiled
```

If `STORED_SPOIL_RATE` is positive, stored berries also lose a whole number
chosen by stochastic rounding. The default stored spoilage rate is zero: sealed
storage keeps berries. With `FEATURES.SPOILAGE: false`, no berries spoil and
storage has no preservation value. With `FEATURES.STORAGE: false` and spoilage
on, every uneaten berry spoils, so a villager can only carry today's food.

Wood wear is a real sink and is charged even when storage is disabled.

## Barter

Barter is optional (`FEATURES.TRADE`). It exchanges whole berries for whole
wood at a whole-berry-per-wood price. There is no money.

Each living agent computes a whole-unit target stock:

```text
berry_target = max(FOOD_PER_ITERATION,
                   min(storage_capacity, RESERVE_DAYS * FOOD_PER_ITERATION))
wood_target  = min(MAX_WOOD_TARGET,
                   ceil(RESERVE_DAYS * FOOD_PER_ITERATION
                        * WOOD_PER_CONTAINER / BERRIES_PER_CONTAINER))
```

`wood_target` is zero unless storage and spoilage are both enabled. When
spoilage is off, the berry target is the full reserve instead of being capped
by storage. When an agent has less than one ration of berries, its
`wood_target` is set to zero: a hungry villager is willing to sell storage wood
to eat.

The agent computes whole surpluses and deficits:

- a berry surplus plus a wood deficit makes it a **berry seller**: it offers
  `min(berry_surplus, wood_deficit * price)` berries, in whole-price bundles;
- a wood surplus plus a berry deficit makes it a **wood seller**: it offers
  `min(wood_surplus, ceil(berry_deficit / price))` wood;
- a wood-rich agent below its own wood target can still sell storage wood to
  cover a berry deficit, especially food.

The market clears at the current whole price:

```text
berry_supply = sum of berry offers
wood_supply  = sum of wood offers
berry_demand = wood_supply * price
berry_traded = min(berry_supply, berry_demand)
wood_traded  = berry_traded / price
```

Sellers are paired at the current price. Every trade moves exactly
`berries = wood * price` from the berry seller to the wood seller and `wood`
in the opposite direction. Both goods are conserved.

The model keeps a fractional `price` internally for smoothing, but all
offers, trades, displays, and metrics use the rounded whole `trade_price`.
The price used for this iteration's offers and trades is the one recorded in
the metrics; the adjusted price is used from the next iteration.

```text
imbalance = (berry_demand - berry_supply) / max(berry_demand, berry_supply)
price = clip(price * (1 - PRICE_ADJUSTMENT * imbalance), MIN_PRICE, MAX_PRICE)
trade_price = clip(round(price), MIN_PRICE, MAX_PRICE)
```

The sign is deliberate: when berries are scarce, one block of wood buys fewer
berries, so the berries-per-wood ratio falls.

## Gifting

Gifting is off by default (`FEATURES.GIFTING: false`). When enabled,
`utils.social.process_berry_gifts` lets a living agent with whole berries above
`MIN_RESERVE` give a whole amount from `AMOUNT_RANGE` to the neediest living
agent, chosen by lowest health and then lowest berries. Gifts move whole
berries directly and conserve the berry total. There is no gift debt,
reciprocity, or trust in the subsistence world yet; those are reserved for
shared social utilities.

## Death

Death is explicit: `health <= 0`. A death is logged once with cause `hunger`.
Dead agents are skipped by every living-agent step. Their goods remain counted
until they wear out or spoil, so a death never silently deletes resources.

## Iteration order

1. Reset per-iteration observations.
2. Regenerate the berry and wood commons as whole units.
3. Allocate the finite day and gather; living agents are processed in random
   order, berries before wood within an agent.
4. Optional need-based whole-berry gifts.
5. Optional whole-unit barter, including the price update.
6. Record each living agent's post-trade holdings.
7. Eat and update health; log any deaths.
8. Apply wood wear and berry spoilage to every agent, including the dead.
9. Clamp stocks to non-negative whole values and health to
   `[0, MAX_HEALTH]`.
10. Record metrics and an agent snapshot.

## Conservation ledger

Let `B = sum(agent.berries) + commons_berries` and
`W = sum(agent.wood) + commons_wood`.

| Process | Berries | Wood | Notes |
| --- | --- | --- | --- |
| Commons regeneration | **source** | **source** | Adds whole units up to `REGENERATION`, capped by `CAPACITY`. |
| Gathering | moves + **sink** | moves + **sink** | Usable goods move from commons to the agent; extraction waste is a sink. |
| Gift | conserved | unchanged | The giver loses exactly what the receiver gains. |
| Barter | conserved | conserved | Whole berries and whole wood move in opposite directions at the agreed ratio. |
| Eating | **sink** | unchanged | Living agents consume up to `FOOD_PER_ITERATION` whole berries. |
| Spoilage | **sink** | unchanged | Whole berries above storage capacity, plus optional stored spoilage. |
| Wood wear | unchanged | **sink** | Whole blocks lost to wear, including on dead agents. |
| Death | unchanged | unchanged | Goods remain in the dead agent's inventory. |

The exact invariants are:

```text
ΔB = regenerated_berries - eaten - spoiled - commons_waste_berries
ΔW = regenerated_wood - wood_decayed - commons_waste_wood
```

`tests/subsistence/test_conservation.py` enforces these every iteration,
including runs with gifting, trade, extraction waste, and deaths.

## Feature flags

Flags live under `FEATURES`:

| Flag | Default | Effect |
| --- | --- | --- |
| `GIFTING` | `false` | Need-based whole-berry gifts from the shared social utility. |
| `TRADE` | `true` | Whole-unit barter at a whole-berry-per-wood price. |
| `STORAGE` | `true` | Wood provides storage capacity. |
| `SPOILAGE` | `true` | Unprotected berries spoil at the end of the iteration. |
| `TRUST` | `false` | Reserved; enabling it currently raises an error. |
| `RECIPROCITY` | `false` | Reserved; enabling it currently raises an error. |
| `SHOCKS` | `false` | Reserved; enabling it currently raises an error. |

The flags are independent. For example:

- `STORAGE: false`, `SPOILAGE: true` — a harsh world where only today's food
  can be carried.
- `SPOILAGE: false` — berries keep indefinitely, so wood has no preservation
  value and barter demand for it disappears.
- `TRADE: false` — self-sufficiency; agents use the priority heuristic and
  surplus time is left idle after their targets are met.
- `GIFTING: true` — the only transfer channel besides barter.

## Replay schema (version 2)

`worlds/subsistence/replay.py` writes and validates schema version 2 with
`meta.world = "subsistence"`.

- `meta` — generator, world, iterations, seed, agent count, feature flags,
  health settings, the finite-day time costs, the storage container ratio,
  resource definitions, initial commons and totals, the initial price, and the
  full config.
- `agents` — id, label, colour, berry and wood skills, the `specialisation_z`
  latent, time budget, time costs, maximum daily yields, initial health, and
  the storage container ratio.
- `ticks` — one row per iteration plus iteration zero. Each tick has summary
  metrics and one state per agent: whole health, berries, wood and storage
  capacity; the labour split; carried wood progress; planned berries and wood;
  actual whole gathered berries and wood; whole post-trade holdings; and alive
  status.
- `events` — typed `gather`, `gift`, `trade`, `spoil`, and `death` records with
  a tick, phase, sequence number, and stable id. Trades carry whole quantities
  and the whole price; gather events carry the whole extracted amount and
  waste.

The viewer reads this replay and never imports the model.

## Metrics

The world records, among others: `alive`, `mean_health`, `total_berries`,
`total_wood`, `commons_berries`, `commons_wood`, `gini_berries`, `gini_wood`,
`barter_price`, `berries_gathered`, `wood_gathered`, `berries_eaten`,
`unmet_food`, `berries_spoiled`, `wood_decayed`, `commons_waste_berries`,
`commons_waste_wood`, `storage_capacity`, `storage_used`,
`storage_utilization`, `trade_berries`, `trade_wood`, `gifts`, `gift_berries`,
`specialisation`, `ppf_efficiency`, and `extinction_iteration`.

`ppf_efficiency` compares the berry value of actual gathering with the berry
value of the planned labour at the current whole price. A value below 1 means
the commons limited the plan.

## Invariants

1. Agent berries, agent wood, the commons stocks, and health never become
   negative, and berries, wood, storage capacity, and health are whole numbers.
2. Health stays within `[0, MAX_HEALTH]`.
3. Labour spent on berries and wood never exceeds the agent's finite day.
4. `wood_progress` is fractional, stays in `[0, 1)`, and carries between
   iterations; it is never counted as stored wood.
5. Total berries and total wood change only by the sources and sinks in the
   ledger above.
5. Gifts and trades conserve both goods.
6. Dead agents do not participate in gathering, gifting, trade, or eating.
7. The barter price stays within `[MIN_PRICE, MAX_PRICE]` and is a whole
   number of berries per wood.
8. Storage capacity is
   `(wood * BERRIES_PER_CONTAINER) // WOOD_PER_CONTAINER` while storage is
   enabled, and zero otherwise.

## Open decisions

- Whether the shared commons should have spatial access or ownership rules.
- Whether wood should have a second use, such as tools or shelter.
- Whether stored berries should spoil slowly by default.
- How to port reciprocity and trust into a shared utility that works for both
  goods and money.
- Whether agents should learn their comparative advantage rather than read the
  current price directly.
