# The Subsistence World

A goods-only world with no money. Villagers start with ten berries and one crate,
spend a finite working day gathering whole berries and raw wood from a shared
commons, craft finished crates, eat, store food, and barter. Health is a short
four-day buffer. Eight raw wood plus a day of skilled carpentry make a crate
that stores ten berries; raw wood itself is only material, not storage.
Crates are perishable and wear faster when full. Gathering and chopping slow
down as the commons thins. Each villager carries an integer price belief, and
bilateral bargaining plus market gossip discovers prices with a one-tick lag.
Villagers keep a small post-meal food reserve and refill it in batches rather
than topping up every day; spare time goes to wood, and owned crates come with a
spare crate's worth of raw wood. Daily choices carry a small taste shock and
occasional exploration. Partial chopping and crafting carry between days. Little
villagers walk out to the berry bushes and forest, visiting several sparse sites
when the commons is depleted.

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
  STORAGE: true
  SPOILAGE: true
```

```bash
python -m worlds.subsistence.experiments --iterations 300 --seeds 1 2 3 \
  --sweep FEATURES.GIFTING=true,false \
  --sweep FEATURES.STORAGE=true,false \
  --output results/subsistence.csv
```

## Files

- `world.py` — agents, commons, labour, health, storage, barter, metrics.
- `config.yaml` — default parameters and flags.
- `replay.py` — version 2 replay builder and validator.
- `visualize.py` — self-contained HTML page builder.
- `viewer/` — canvas village, event feed, inspector, and PPF chart.
- `experiments.py` — seed and flag sweeps.
- `MODEL.md` — full rules, formulas, conservation ledger, and replay schema.

The shared config, metrics, replay I/O, experiment, and social helpers live in
`../../utils/`.
