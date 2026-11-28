# The Subsistence World

A goods-only world with no money. Villagers start with nothing, spend a finite
working day gathering whole berries and whole wood from a shared commons, eat,
store food, and optionally barter. Health is a short four-day buffer, eight
wood blocks store ten berries, chopping is much slower than picking, and a
shared specialisation latent makes berry and wood skills negatively correlated.
Partial chopping carries between days. Little villagers walk out to the berry
bushes and forest during the run.

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
