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
- `berries`, `wood`, and `crates`;
- `price_belief` — whole berries per wood, used for planning and barter;
- `reserve_days_preference` — this villager's stable taste for a food reserve;
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

Berries regrow quickly and are abundant. Wood regrows much more slowly and is
the scarce, valuable resource. Density is measured against the initial stock
and capped at 1, so a stock above its initial level is still treated as full;
the `MIN_DENSITY` floor of `0.10` prevents division by zero when a stock is
exhausted. Each iteration records `berries_commons_pct` and
`wood_commons_pct` for the harvesting phase, and the viewer uses those values
for both the frontier and the multi-site search animation.

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

With two days and a six-berry ration the target is twelve berries. One crate
therefore holds a full reserve, while two crates hold twelve of their twenty
slots: the second crate is useful without being filled to the brim. A small
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
Each villager has an integer `price_belief` in berries per wood, initialized
near the village baseline opportunity cost with a small random offset and kept
inside a configurable band around that baseline.

### Planning with beliefs

A villager plans the day using its own `price_belief` and the latest observed
commons density. The wood-versus-berries choice uses:

```text
berry value/minute = berry_skill * density / BERRY_TIME_COST
wood value/minute  = wood_skill * density * price_belief / WOOD_TIME_COST
```

Each day the villager draws a log-normal taste shock on each value and, with
probability `EXPLORATION_RATE`, swaps the two activities. This is deliberate
sampling rather than a deterministic argmax: choices vary slightly around the
rational comparison, and a villager occasionally tries the other activity.

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
wood seller min    = max(blend(opportunity, belief), subsistence) + urgency
wood buyer max     = blend(opportunity, belief) + storage urgency
crate seller min   = blend(crate opportunity, crate belief) + urgency
crate buyer max    = blend(crate opportunity, crate belief) + urgency
```

The blend is mostly belief with a smaller opportunity-cost anchor. A wood
buyer may spend saved berries down to today's meal, so savings can smooth a thin
market instead of limiting bids to the day's production surplus. A seller whose
health is low and who is short of food accepts almost any price, because eating
today matters more than the quoted ratio. Crate opportunities include the eight
raw wood and the craft minutes valued at the villager's berry rate.

### Bilateral matching

The market square runs two order books, crates first and then raw wood:

- sellers have a private minimum acceptable price and a whole-unit quantity;
- buyers have a private maximum willingness to pay, a quantity, and a berry
  budget;
- the cheapest asks are matched against the highest bids first, with a random
  tie-break, so a desperate seller reaches the buyers who can actually pay.

If `seller_min <= buyer_max`, the whole-unit trade clears at the integer
midpoint:

```text
trade_price = (seller_min + buyer_max) // 2
```

Every trade moves whole goods and conserves them. Wood trades move
`berries = wood * price`; crate trades move `berries = crates * crate_price`.
The market price recorded in the header is the realized average integer price
for the iteration, or the last observed market price when no trade occurred.

### Information lag and gossip

All reservations and trades use the start-of-tick beliefs. Belief updates are
batched until the barter phase ends, so they take effect on tick `t + 1`:

- traders move one berry toward their realized trade price;
- other villagers who actually entered the market square move one gossip step
  toward the average trade price;
- a fully unmatched seller lowers its belief by one; a fully unmatched buyer
  raises it by one, within the belief band and `[MIN_PRICE, MAX_PRICE]`.

Villagers who did not enter the market do not receive gossip and do not change
their beliefs.

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
2. Regenerate the berry and wood commons as whole units, then record
   `berries_commons_pct` and `wood_commons_pct` for the harvest.
3. Allocate the finite day using density-adjusted costs and gather; living
   agents are processed in random order, berries before wood within an agent.
4. Optional need-based whole-berry gifts.
5. Optional bilateral barter: crates first, then raw wood, using start-of-tick
   beliefs; batch all belief and gossip updates for tick `t + 1`.
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
| Barter | conserved | conserved | Whole berries move against whole wood or finished crates at agreed prices. |
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
including runs with gifting, trade, extraction waste, and deaths.

## Feature flags

Flags live under `FEATURES`:

| Flag | Default | Effect |
| --- | --- | --- |
| `GIFTING` | `false` | Need-based whole-berry gifts from the shared social utility. |
| `TRADE` | `true` | Bilateral whole-unit barter in wood and finished crates at decentralized prices. |
| `STORAGE` | `true` | Finished crates provide storage capacity. |
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
  health settings, finite-day time costs, crate crafting and wear settings,
  belief bounds, resource definitions, initial commons and totals, the initial
  market price, and the full config.
- `agents` — id, label, colour, berry and wood skills, the `specialisation_z`
  latent, initial price belief, time budget, time costs, craft minutes, base
  maximum daily yields, initial health, initial crates, and the storage
  container ratio.
- `ticks` — one row per iteration plus iteration zero. Each tick has summary
  metrics, including commons density, market price, beliefs, crate counts, and
  crafting totals, and one state per agent: whole health, berries, raw wood,
  crates with durability, storage capacity, crafting progress, price belief,
  the berry/wood/craft labour split, density-adjusted effective yields, carried
  wood progress, planned goods, actual gathered goods, post-trade holdings, and
  alive status.
- `events` — typed `gather`, `craft`, `gift`, `trade`, `spoil`, and `death`
  records with a tick, phase, sequence number, and stable id. Trades carry the
  goods type, whole quantities, and whole price; craft events carry the raw
  wood consumed.

The viewer reads this replay and never imports the model.

## Metrics

The world records, among others: `alive`, `mean_health`, `total_berries`,
`total_wood`, `total_crates`, `crate_durability`, `crates_crafted`,
`crates_worn`, `wood_crafted`, `commons_berries`, `commons_wood`,
`berries_commons_pct`, `wood_commons_pct`, `gini_berries`, `gini_wood`,
`barter_price`, `last_crate_price`, `mean_price_belief`, `market_participants`,
`market_traders`, `failed_trades`, `berries_gathered`, `wood_gathered`,
`berries_eaten`, `unmet_food`, `berries_spoiled`, `wood_decayed`,
`commons_waste_berries`, `commons_waste_wood`, `storage_capacity`,
`storage_used`, `storage_utilization`, `trade_berries`, `trade_wood`,
`crate_trades`, `trade_crates`, `gifts`, `gift_berries`, `specialisation`,
`ppf_efficiency`, and `extinction_iteration`.

`ppf_efficiency` compares the berry value of actual gathering with the berry
value of the planned labour at the current whole price. A value below 1 means
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
7. Gifts and trades conserve both goods; crate trades conserve crates and
   berries.
8. Dead agents do not participate in gathering, crafting, gifting, trade, or
   eating.
9. Every agent's price belief stays within `[MIN_PRICE, MAX_PRICE]` and within
   the configured baseline band. Trade prices are whole berries per unit.
10. Storage capacity is `crate_count * BERRIES_PER_CONTAINER` while storage is
    enabled, and zero otherwise.
11. Crate durability stays in `(0, 1]`; crates at zero durability are removed.

## Open decisions

- Whether the shared commons should have spatial access or ownership rules.
- Whether wood should have a second use, such as tools or shelter.
- Whether stored berries should spoil slowly by default.
- How to port reciprocity and trust into a shared utility that works for both
  goods and money.
- Whether agents should learn their comparative advantage rather than read
  their current price belief directly.
- How crate durability should vary by material or craft skill.
