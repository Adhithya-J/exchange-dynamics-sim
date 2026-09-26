"""A small agent-based simulation of resource sharing.

The simulation keeps agents as ordinary Python objects. Pandas is used only
for presenting the current state and calculating summary statistics.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from random import Random
from typing import Any, Iterable

import numpy as np
import pandas as pd


CONFIG: dict[str, dict[str, Any]] = {
    "ENV_INIT": {
        "N_AGENTS": 10,
        "N_ITERATIONS": 5_000,
        "MAX_RESOURCES": 1_000.0,
        "MIN_RESOURCES": 0.0,
        "SEED": 42,
        "RESOURCE_TRANSFER_SIZE": 1.0,
    },
    "AGENTS_INIT": {
        "GENEROSITY_RANGE": (0.0, 1.0),
        "ACCEPTANCE_RANGE": (0.0, 1.0),
        "GIVING_FLOOR": 1.0,
        "RECEIVING_CEIL": 10_000.0,
        "COST_OF_LIVING": 0.001,
    },
    "AFFORDABILITY": {
        "RESOURCE_MIN": 50.0,
        "RESOURCE_MAX": 300.0,
        "LOWER_LIMIT": 0.0,
        "UPPER_LIMIT": 1.0,
    },
    "MEMORY": {
        "MEMORY_SIZE": 10,
        "DEFAULT_WEIGHT": 1.0,
        "MEMORY_BONUS": 3.0,
    },
}

# Keep the old lowercase name as a small compatibility convenience for users
# who imported the original module-level configuration.
config = CONFIG


class Agent:
    """An individual with resources, social tendencies, and recipient memory."""

    def __init__(
        self,
        id: str,
        generosity_score: float,
        acceptance_score: float,
        initial_resources: float,
        memory_size: int = 10,
    ) -> None:
        self.id = id
        self.generosity_score = float(generosity_score)
        self.acceptance_score = float(acceptance_score)
        self.resources = float(initial_resources)
        self.memory: deque[str] = deque(maxlen=memory_size)

    @property
    def is_alive(self) -> bool:
        return self.resources > 0

    def can_give(self, giving_floor: float, transfer_size: float) -> bool:
        """Return whether a transfer can leave the agent above the giving floor."""

        return (
            self.resources >= transfer_size
            and self.resources - transfer_size >= giving_floor
        )

    def can_receive(self, receiving_ceil: float, transfer_size: float) -> bool:
        """Return whether a transfer would keep the agent below its ceiling."""

        return self.resources + transfer_size <= receiving_ceil

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the agent."""

        return {
            "id": self.id,
            "generosity_score": self.generosity_score,
            "acceptance_score": self.acceptance_score,
            "resources": self.resources,
            "memory": list(self.memory),
        }


class AffordabilityCalculator:
    """Convert resources into a normalized affordability score."""

    def __init__(self, config: dict[str, dict[str, Any]]) -> None:
        self.config = config

    def affordability(self, resources: float) -> float:
        low = self.config["AFFORDABILITY"]["RESOURCE_MIN"]
        high = self.config["AFFORDABILITY"]["RESOURCE_MAX"]
        if high <= low:
            raise ValueError("RESOURCE_MAX must be greater than RESOURCE_MIN")
        if resources <= low:
            return self.config["AFFORDABILITY"]["LOWER_LIMIT"]
        if resources >= high:
            return self.config["AFFORDABILITY"]["UPPER_LIMIT"]
        return (resources - low) / (high - low)

    def effective_generosity(self, agent: Agent) -> float:
        score = agent.generosity_score * self.affordability(agent.resources)
        lower_limit = self.config["AGENTS_INIT"]["GENEROSITY_RANGE"][0]
        upper_limit = self.config["AGENTS_INIT"]["GENEROSITY_RANGE"][1]
        return float(np.clip(score, lower_limit, upper_limit))

    def effective_acceptance(self, agent: Agent) -> float:
        score = agent.acceptance_score * (2 - self.affordability(agent.resources))
        lower_limit = self.config["AGENTS_INIT"]["ACCEPTANCE_RANGE"][0]
        upper_limit = self.config["AGENTS_INIT"]["ACCEPTANCE_RANGE"][1]
        return float(np.clip(score, lower_limit, upper_limit))


class MetricsCalculator:
    """Calculate inequality and resource statistics."""

    @staticmethod
    def calculate_gini(values: Iterable[float]) -> float:
        values_array = np.asarray(list(values), dtype=float)
        if values_array.size == 0:
            raise ValueError("Input array must not be empty")
        if np.any(values_array < 0):
            raise ValueError("Gini is undefined for negative values")

        total = values_array.sum()
        if total == 0:
            return 0.0

        differences = np.abs(values_array[:, None] - values_array[None, :])
        return float(differences.sum() / (2 * values_array.size * total))

    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> dict[str, float]:
        if "resources" not in df:
            raise ValueError("DataFrame must contain a resources column")
        resources = df["resources"].to_numpy(dtype=float)
        if resources.size == 0:
            raise ValueError("Cannot calculate statistics for an empty DataFrame")
        return {
            "gini": MetricsCalculator.calculate_gini(resources),
            "mean": float(np.mean(resources)),
            "median": float(np.median(resources)),
            "std": float(np.std(resources)),
            "min": float(np.min(resources)),
            "max": float(np.max(resources)),
            "total": float(np.sum(resources)),
        }


class ResourceSimulation:
    """Run resource transfers between a collection of agents."""

    def __init__(self, config: dict[str, dict[str, Any]] | None = None) -> None:
        self.config = deepcopy(config or CONFIG)
        self.random = Random(self.config["ENV_INIT"]["SEED"])
        self.affordability = AffordabilityCalculator(self.config)
        self.agents: list[Agent] = []
        self.metrics_history: list[dict[str, float]] = []

    def initialize_agents(self) -> list[Agent]:
        """Create the initial population using the configured random seed."""

        generosity_range = self.config["AGENTS_INIT"]["GENEROSITY_RANGE"]
        acceptance_range = self.config["AGENTS_INIT"]["ACCEPTANCE_RANGE"]
        memory_size = self.config["MEMORY"]["MEMORY_SIZE"]
        initial_resources = self.config["ENV_INIT"]["MAX_RESOURCES"]
        count = self.config["ENV_INIT"]["N_AGENTS"]

        self.agents = [
            Agent(
                id=f"agent-{index + 1:03d}",
                generosity_score=self.random.uniform(*generosity_range),
                acceptance_score=self.random.uniform(*acceptance_range),
                initial_resources=initial_resources,
                memory_size=memory_size,
            )
            for index in range(count)
        ]
        self.metrics_history = []
        return self.agents

    def _agent_by_id(self, agent_id: str) -> Agent:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise KeyError(f"Unknown agent: {agent_id}")

    def _find_receiver(self, sender: Agent) -> Agent | None:
        transfer_size = self.config["ENV_INIT"]["RESOURCE_TRANSFER_SIZE"]
        receiving_ceil = self.config["AGENTS_INIT"]["RECEIVING_CEIL"]
        candidates = [
            agent
            for agent in self.agents
            if agent.id != sender.id
            and agent.can_receive(receiving_ceil, transfer_size)
        ]
        if not candidates:
            return None

        default_weight = self.config["MEMORY"]["DEFAULT_WEIGHT"]
        memory_bonus = self.config["MEMORY"]["MEMORY_BONUS"]
        remembered = list(sender.memory)
        weights = []
        for candidate in candidates:
            weight = default_weight
            if candidate.id in remembered:
                # More recent remembered givers receive a larger preference.
                recency = remembered.index(candidate.id) + 1
                weight += memory_bonus * recency / len(remembered)
            weights.append(weight)
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _can_give(self, agent: Agent) -> bool:
        transfer_size = self.config["ENV_INIT"]["RESOURCE_TRANSFER_SIZE"]
        giving_floor = self.config["AGENTS_INIT"]["GIVING_FLOOR"]
        return (
            agent.can_give(giving_floor, transfer_size)
            and self.random.random() < self.affordability.effective_generosity(agent)
        )

    def _can_receive(self, agent: Agent) -> bool:
        transfer_size = self.config["ENV_INIT"]["RESOURCE_TRANSFER_SIZE"]
        receiving_ceil = self.config["AGENTS_INIT"]["RECEIVING_CEIL"]
        return (
            agent.can_receive(receiving_ceil, transfer_size)
            and self.random.random() < self.affordability.effective_acceptance(agent)
        )

    def _generate_transfer_actions(self) -> list[tuple[str, str]]:
        actions = []
        for sender in self.agents:
            if not self._can_give(sender):
                continue
            receiver = self._find_receiver(sender)
            if receiver is not None and self._can_receive(receiver):
                actions.append((sender.id, receiver.id))
        return actions

    def _perform_transfers(
        self, actions: Iterable[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        transfer_size = self.config["ENV_INIT"]["RESOURCE_TRANSFER_SIZE"]
        giving_floor = self.config["AGENTS_INIT"]["GIVING_FLOOR"]
        receiving_ceil = self.config["AGENTS_INIT"]["RECEIVING_CEIL"]
        completed = []

        for giver_id, receiver_id in actions:
            giver = self._agent_by_id(giver_id)
            receiver = self._agent_by_id(receiver_id)
            if not giver.can_give(giving_floor, transfer_size):
                continue
            if not receiver.can_receive(receiving_ceil, transfer_size):
                continue
            giver.resources -= transfer_size
            receiver.resources += transfer_size
            receiver.memory.append(giver.id)
            completed.append((giver.id, receiver.id))
        return completed

    def _apply_living_cost(self) -> None:
        cost = self.config["AGENTS_INIT"]["COST_OF_LIVING"]
        minimum = self.config["ENV_INIT"]["MIN_RESOURCES"]
        for agent in self.agents:
            agent.resources = max(minimum, agent.resources * (1 - cost))

    def to_frame(self) -> pd.DataFrame:
        """Return the current agent state as a dataframe for analysis."""

        columns = [
            "id",
            "generosity_score",
            "acceptance_score",
            "resources",
            "memory",
        ]
        if not self.agents:
            return pd.DataFrame(columns=columns).set_index("id")
        return pd.DataFrame([agent.to_dict() for agent in self.agents]).set_index("id")

    def run_iteration(self) -> list[tuple[str, str]]:
        """Run one transfer-and-living-cost step and return completed transfers."""

        if not self.agents:
            self.initialize_agents()
        actions = self._generate_transfer_actions()
        completed = self._perform_transfers(actions)
        self._apply_living_cost()
        self.metrics_history.append(MetricsCalculator.calculate_statistics(self.to_frame()))
        return completed

    def run(self, iterations: int | None = None) -> pd.DataFrame:
        """Run the configured number of iterations and return the final state."""

        if not self.agents:
            self.initialize_agents()
        count = self.config["ENV_INIT"]["N_ITERATIONS"] if iterations is None else iterations
        if count < 0:
            raise ValueError("iterations must not be negative")
        for _ in range(count):
            self.run_iteration()
        return self.to_frame()


def main() -> None:
    simulation = ResourceSimulation()
    final_state = simulation.run()
    print(final_state.head())
    print("\nFinal statistics:")
    print(simulation.metrics_history[-1])


if __name__ == "__main__":
    main()
