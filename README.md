# Exchange Dynamics Simulation

An experimental agent-based simulation of resource sharing and inequality.
Agents have different generosity and acceptance tendencies. During each
iteration, living agents may transfer a variable amount of resources to another
living agent, remember previous givers, and pay a proportional cost of living.
An agent that reaches the configured resource floor is dead and no longer
participates in transfers.

## Status

This is a small research-style simulation project. It is intended to make the
rules and measurements easy to inspect, not to model a real economy or make
economic predictions.

Supported Python versions: 3.11 through 3.14.

## Model

Each agent has:

- a resource balance;
- a resource floor below which the agent is dead;
- a generosity score, which controls the chance of giving;
- an acceptance score, which controls the chance of receiving; and
- short-term memory of agents that previously transferred resources to them.

Affordability is derived from the agent's resource balance. Agents below the
configured resource range become less likely to give, while acceptance is
weighted in the opposite direction. Transfer amounts are sampled from the
configured range and capped by the giving floor and receiving ceiling. A
transfer is made only when both agents are alive. After transfers, every
living agent pays the configured cost of living; reaching the resource floor
does not create a resource top-up.

The simulation records mean, median, standard deviation, total resources, and
the Gini coefficient after each iteration. The source of truth is a list of
`Agent` objects; Pandas is used only to expose the current state for analysis.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -r requirements.txt
python main.py
python visualize.py --iterations 500
python -m pytest -q
```

`visualize.py` creates a self-contained `simulation.html` Plotly chart. Open it
in a browser to inspect each agent's resource balance over time and move the
iteration slider. The summary card in the chart updates with the selected
iteration.

The random seed and model parameters are defined in `config.yaml` and loaded
through `utils.load_config`. For interactive use, construct
`ResourceSimulation` with a copied and edited configuration, call `run()`, and
inspect `metrics_history` or the returned DataFrame.

## Project structure

- `config.yaml` - default simulation parameters.
- `main.py` - agents, simulation lifecycle, affordability, and metrics.
- `utils/` - shared helpers, including YAML configuration loading.
- `tests/` - unit and behavior tests for the model.
- `visualize.py` - interactive Plotly visualization.
- `requirements.txt` - runtime and test dependencies.

## Limitations

The model has no production data, wages, rent, births, deaths, or spatial
network. Its results depend on the chosen rules and random seed. It should be
treated as an educational simulation and a base for further experiments.
