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

from utils import load_config

CONFIG: dict[str, dict[str, Any]] = load_config()

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
        resource_floor: float = 0.0,
    ) -> None:
        self.id = id
        self.generosity_score = float(generosity_score)
        self.acceptance_score = float(acceptance_score)
        self.resources = float(initial_resources)
        self.resource_floor = float(resource_floor)
        self.memory: deque[str] = deque(maxlen=memory_size)

    @property
    def is_alive(self) -> bool:
        """Whether the agent has resources strictly above the death floor."""

        return self.resources > self.resource_floor

    def can_give(self, giving_floor: float, transfer_size: float) -> bool:
        """Return whether a transfer can leave the agent above the giving floor."""

        return (
            self.is_alive
            and self.resources >= transfer_size
            and self.resources - transfer_size >= giving_floor
        )

    def can_receive(self, receiving_ceil: float, transfer_size: float) -> bool:
        """Return whether a transfer would keep the agent below its ceiling."""

        return self.is_alive and self.resources + transfer_size <= receiving_ceil

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the agent."""

        return {
            "id": self.id,
            "generosity_score": self.generosity_score,
            "acceptance_score": self.acceptance_score,
            "resources": self.resources,
            "alive": self.is_alive,
            "memory": list(self.memory),
        }


class AffordabilityCalculator:
    """Convert resources into a normalized affordability score."""

    def __init__(self, config: dict[str, dict[str, Any]]) -> None:
        self.config = config

    def affordability(self, resources: float) -> float:
        low = self.config["AFFORDABILITY"]["MIN_RESOURCES"]
        high = self.config["AFFORDABILITY"]["MAX_RESOURCES"]
        if high <= low:
            raise ValueError("MAX_RESOURCES must be greater than MIN_RESOURCES")
        if resources <= low:
            return self.config["AFFORDABILITY"]["MIN_SCORE"]
        if resources >= high:
            return self.config["AFFORDABILITY"]["MAX_SCORE"]
        return (resources - low) / (high - low)

    def effective_generosity(self, agent: Agent) -> float:
        score = agent.generosity_score * self.affordability(agent.resources)
        lower_limit = self.config["AGENTS"]["GENEROSITY"][0]
        upper_limit = self.config["AGENTS"]["GENEROSITY"][1]
        return float(np.clip(score, lower_limit, upper_limit))

    def effective_acceptance(self, agent: Agent) -> float:
        score = agent.acceptance_score * (1.0 - self.affordability(agent.resources))
        lower_limit = self.config["AGENTS"]["ACCEPTANCE"][0]
        upper_limit = self.config["AGENTS"]["ACCEPTANCE"][1]
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
        self.random = Random(self.config["SIMULATION"]["RANDOM_SEED"])
        self.affordability = AffordabilityCalculator(self.config)
        self.agents: list[Agent] = []
        self.metrics_history: list[dict[str, Any]] = []
        self.agent_history: list[dict[str, Any]] = []
        self.total_transfers = 0
        self.total_transfer_volume = 0.0

    def initialize_agents(self) -> list[Agent]:
        """Create the initial population using the configured random seed."""

        generosity_range = self.config["AGENTS"]["GENEROSITY"]
        acceptance_range = self.config["AGENTS"]["ACCEPTANCE"]
        memory_size = self.config["MEMORY"]["SIZE"]
        initial_resources = self.config["SIMULATION"]["INITIAL_RESOURCES"]
        resource_floor = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        count = self.config["SIMULATION"]["AGENTS"]

        self.agents = [
            Agent(
                id=f"agent-{index + 1:03d}",
                generosity_score=self.random.uniform(*generosity_range),
                acceptance_score=self.random.uniform(*acceptance_range),
                initial_resources=initial_resources,
                memory_size=memory_size,
                resource_floor=resource_floor,
            )
            for index in range(count)
        ]
        self.metrics_history = []
        self.agent_history = []
        self.total_transfers = 0
        self.total_transfer_volume = 0.0
        self._record_agent_snapshot(iteration=0)
        return self.agents

    def _record_agent_snapshot(self, iteration: int) -> None:
        """Record the state needed to visualize every agent over time."""

        for agent in self.agents:
            self.agent_history.append(
                {
                    "iteration": iteration,
                    "id": agent.id,
                    "resources": agent.resources,
                    "generosity_score": agent.generosity_score,
                    "acceptance_score": agent.acceptance_score,
                }
            )

    def _agent_by_id(self, agent_id: str) -> Agent:
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        raise KeyError(f"Unknown agent: {agent_id}")

    def _transfer_amount_bounds(self) -> tuple[float, float]:
        """Return the configured minimum and maximum transfer amounts."""

        simulation_settings = self.config["SIMULATION"]
        if "TRANSFER_AMOUNT_RANGE" in simulation_settings:
            lower, upper = simulation_settings["TRANSFER_AMOUNT_RANGE"]
        else:
            # Keep compatibility with configurations that still use the old
            # fixed transfer amount.
            lower = upper = simulation_settings["TRANSFER_AMOUNT"]
        lower = float(lower)
        upper = float(upper)
        if lower <= 0 or upper < lower:
            raise ValueError(
                "TRANSFER_AMOUNT_RANGE must contain positive values in ascending order"
            )
        return lower, upper

    def _find_receiver(
        self, sender: Agent, minimum_transfer: float | None = None
    ) -> Agent | None:
        if minimum_transfer is None:
            minimum_transfer, _ = self._transfer_amount_bounds()
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        candidates = [
            agent
            for agent in self.agents
            if agent.id != sender.id
            and agent.can_receive(receiving_ceil, minimum_transfer)
        ]
        if not candidates:
            return None

        memory_fraction = self.config["MEMORY"]["PREFERENCE_FRACTION"]
        if not 0 <= memory_fraction <= 1:
            raise ValueError("PREFERENCE_FRACTION must be between 0 and 1")

        remembered_candidates = [
            candidate for candidate in candidates if candidate.id in sender.memory
        ]
        if not remembered_candidates or memory_fraction == 0:
            return self.random.choice(candidates)

        candidate_memory_scores: dict[str, float] = {}
        for idx, agent_id in enumerate(sender.memory):
            candidate_memory_scores[agent_id] = (
                candidate_memory_scores.get(agent_id, 0.0) + (idx + 1)
            )

        total_memory_score = sum(
            candidate_memory_scores[candidate.id]
            for candidate in remembered_candidates
        )

        uniform_share = (1 - memory_fraction) / len(candidates)
        weights = []
        for candidate in candidates:
            weight = uniform_share
            if candidate.id in candidate_memory_scores:
                memory_share = candidate_memory_scores[candidate.id] / total_memory_score
                weight += memory_fraction * memory_share
            weights.append(weight)
        return self.random.choices(candidates, weights=weights, k=1)[0]

    def _can_give(self, agent: Agent, transfer_size: float) -> bool:
        giving_floor = self.config["AGENTS"]["MIN_RESOURCES_AFTER_GIVING"]
        return (
            agent.can_give(giving_floor, transfer_size)
            and self.random.random() < self.affordability.effective_generosity(agent)
        )

    def _can_receive(self, agent: Agent, transfer_size: float) -> bool:
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        return (
            agent.can_receive(receiving_ceil, transfer_size)
            and self.random.random() < self.affordability.effective_acceptance(agent)
        )

    def _generate_transfer_actions(self) -> list[tuple[str, str, float]]:
        minimum_transfer, maximum_transfer = self._transfer_amount_bounds()
        giving_floor = self.config["AGENTS"]["MIN_RESOURCES_AFTER_GIVING"]
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        actions = []
        for sender in self.agents:
            if not sender.is_alive or not self._can_give(sender, minimum_transfer):
                continue
            receiver = self._find_receiver(sender, minimum_transfer)
            if receiver is None:
                continue

            feasible_maximum = min(
                maximum_transfer,
                sender.resources - giving_floor,
                receiving_ceil - receiver.resources,
            )
            if feasible_maximum < minimum_transfer:
                continue
            transfer_size = self.random.uniform(minimum_transfer, feasible_maximum)
            if self._can_receive(receiver, transfer_size):
                actions.append((sender.id, receiver.id, transfer_size))
        return actions

    def _perform_transfers(
        self, actions: Iterable[tuple[str, str, float]]
    ) -> list[tuple[str, str, float]]:
        giving_floor = self.config["AGENTS"]["MIN_RESOURCES_AFTER_GIVING"]
        receiving_ceil = self.config["AGENTS"]["MAX_RESOURCES_AFTER_RECEIVING"]
        completed = []

        for giver_id, receiver_id, transfer_size in actions:
            giver = self._agent_by_id(giver_id)
            receiver = self._agent_by_id(receiver_id)
            if not giver.can_give(giving_floor, transfer_size):
                continue
            if not receiver.can_receive(receiving_ceil, transfer_size):
                continue
            giver.resources -= transfer_size
            receiver.resources += transfer_size
            receiver.memory.append(giver.id)
            completed.append((giver.id, receiver.id, transfer_size))
        return completed

    def _apply_living_cost(self) -> None:
        cost = self.config["AGENTS"]["LIVING_COST"]
        minimum = self.config["SIMULATION"]["RESOURCE_FLOOR"]
        for agent in self.agents:
            if not agent.is_alive:
                continue
            agent.resources -= cost
            if agent.resources <= minimum:
                # Reaching the floor is death, not a resource top-up.
                agent.resources = minimum

    def to_frame(self) -> pd.DataFrame:
        """Return the current agent state as a dataframe for analysis."""

        columns = [
            "id",
            "generosity_score",
            "acceptance_score",
            "resources",
            "alive",
            "memory",
        ]
        if not self.agents:
            return pd.DataFrame(columns=columns).set_index("id")
        return pd.DataFrame([agent.to_dict() for agent in self.agents]).set_index("id")

    def agent_history_frame(self) -> pd.DataFrame:
        """Return one row per agent and iteration for visualization."""

        columns = [
            "iteration",
            "id",
            "resources",
            "generosity_score",
            "acceptance_score",
        ]
        return pd.DataFrame(self.agent_history, columns=columns)

    def run_iteration(self) -> list[tuple[str, str, float]]:
        """Run one transfer-and-living-cost step and return completed transfers."""

        if not self.agents:
            self.initialize_agents()
        actions = self._generate_transfer_actions()
        completed = self._perform_transfers(actions)
        self._apply_living_cost()
        iteration = len(self.metrics_history) + 1
        self.total_transfers += len(completed)
        transfer_volume = sum(amount for _, _, amount in completed)
        self.total_transfer_volume += transfer_volume
        statistics = MetricsCalculator.calculate_statistics(self.to_frame())
        statistics.update(
            {
                "iteration": iteration,
                "transfers": len(completed),
                "total_transfers": self.total_transfers,
                "transfer_volume": transfer_volume,
                "total_transfer_volume": self.total_transfer_volume,
            }
        )
        self.metrics_history.append(statistics)
        self._record_agent_snapshot(iteration)
        return completed

    def run(self, iterations: int | None = None) -> pd.DataFrame:
        """Run the configured number of iterations and return the final state."""

        if not self.agents:
            self.initialize_agents()
        count = self.config["SIMULATION"]["ITERATIONS"] if iterations is None else iterations
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
