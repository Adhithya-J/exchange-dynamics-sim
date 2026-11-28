# The Subsistence World

A goods-only world with no money. Villagers start with ten berries and one crate,
spend a finite working day gathering whole berries and raw wood from a shared
commons, craft finished crates, eat, store food, and barter. Health is a short
four-day buffer. Eight raw wood plus a day of skilled carpentry make a crate
that stores ten berries; raw wood itself is only material, not storage.
Crates are perishable and wear faster when full. Gathering and chopping slow
down as the commons thins. Each villager carries a fractional price belief, and
bilateral bargaining plus market gossip discovers prices with a one-tick lag.
Prices float freely; the goods exchanged stay whole, so rounding the berry leg is
an explicit indivisibility cost.
Villagers keep a small post-meal food reserve and refill it in batches rather
than topping up every day; spare time goes to wood, and owned crates come with a
spare crate's worth of raw wood. Daily choices carry a small taste shock and
occasional exploration. Optional layers add credit in two modes, reciprocal
gifts, public trust, and commons shocks. Credit can move wood, berries, or crate
capacity; reciprocal loans are zero-interest and gated by trust or collateral,
while market loans price default risk and crate durability into a goods interest
rate. Food is the first loan priority, berry debt keeps today's meal, and formal
loans settle before soft gifts. Partial chopping and crafting carry between days.
Little villagers walk out to the berry bushes and forest, visiting several sparse
sites when the commons is depleted.

Seasons turn every 45 iterations by default: berries regrow fastest in summer
and not at all in winter, while wood keeps regrowing all year. The commons is
the village granary, so a village that strips the bushes in summer goes hungry
in winter. Commons stocks and regeneration also scale with the number of agents
relative to a 10-agent baseline, so 50- and 100-agent runs keep a comparable
per-villager ecology.

A 50-villager, 5,000-iteration run is recorded in
[`../../docs/EMERGENT_FINDINGS.md`](../../docs/EMERGENT_FINDINGS.md), with a
60-second video of one full year and the measured cold-start, winter, crate, and
division-of-labour findings.

## Run it

```bash
# Run and write a standalone viewer page.
python -m worlds.subsistence --iterations 500 \
  --replay replay-subsistence.json.gz \
  --html simulation-subsistence.html

# Run and print the final state only.
python -m worlds.subsistence --iterations 500

# Write only a replay.
python -m worlds.subsistence --iterations 1000 \
  --replay replay-subsistence.json.gz
```

Open `simulation-subsistence.html` in a browser, or serve the repository root
and open:

```text
http://localhost:8000/worlds/subsistence/viewer/?replay=/replay-subsistence.json.gz
```

## Flags

Edit `config.yaml` or sweep the flags in an experiment:

```yaml
FEATURES:
  GIFTING: false
  TRADE: true
  CREDIT: false
  STORAGE: true
  SPOILAGE: true
  TRUST: false
  RECIPROCITY: false
  SHOCKS: false
```

`RECIPROCITY` automatically enables `GIFTING` and `TRUST`; `CREDIT`
automatically enables `TRUST`. Set `CREDIT.MODE` to `reciprocal` for
zero-interest, trust-or-collateral loans, or `market` for risk-priced goods
interest.

```bash
python -m worlds.subsistence.experiments --iterations 300 --seeds 1 2 3 \
  --sweep FEATURES.GIFTING=true,false \
  --sweep FEATURES.STORAGE=true,false \
  --sweep FEATURES.CREDIT=true \
  --sweep CREDIT.MODE=reciprocal,market \
  --output results/subsistence.csv
```

The experiment summary sums per-tick flow metrics over the run and leaves stock
metrics at the final tick. Confidence intervals use a Student-t critical value,
and a single-seed group is reported with an undefined interval.

## Files

- `world.py` — agents, commons, labour, health, storage, barter, metrics, and
  the feature registry.
- `features/` — composable layers for trade, gifting, trust, reciprocity,
  credit, and shocks, plus the shared goods ledger and obligation book.
- `config.yaml` — default parameters and flags.
- `replay.py` — version 3 replay builder and validator (reads version 2 too).
- `visualize.py` — self-contained HTML page builder.
- `viewer/` — canvas village, event feed, inspector, and PPF chart.
- `experiments.py` — seed and flag sweeps.
- `MODEL.md` — full rules, formulas, conservation ledger, and replay schema.

The shared config, metrics, replay I/O, experiment, and social helpers live in
`../../utils/`.
