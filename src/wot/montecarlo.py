import numpy as np
import multiprocessing as mp

from copy import deepcopy
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Callable


@dataclass(kw_only=True)
class SimulationState:
    received_vehicles: dict[str, int] = field(default_factory=dict)
    opened_containers: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pity_counter: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    received_containers: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def increment_container(self, variant: str) -> None:
        if variant not in self.opened_containers:
            self.opened_containers[variant] = 0
        self.opened_containers[variant] += 1


@dataclass(kw_only=True)
class VariantConfig:
    name: str
    vehicle_probability: list[float]
    possible_vehicles: int
    container_probability: float
    pity_threshold: int

@dataclass(kw_only=True)
class SimulationConfig:
    variants: dict[str, VariantConfig] = field(default_factory=dict)
    intervariant_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    preowned_vehicles: dict[str, int] = field(default_factory=dict)

    def possible_vehicles(self, state: SimulationState) -> dict[str, list[str]]:
        vehicles = {}
        for name, variant in self.variants.items():
            vehicles[name] = variant.possible_vehicles - state.received_vehicles.get(name, 0) - self.preowned_vehicles.get(name, 0)
        return vehicles
    

def montecarlo_for_target(config: SimulationConfig, target_fn: Callable[[SimulationState], bool]) -> SimulationState:
    state = SimulationState()
    container_deque = deque(["proto"])
    possible_vehicles = config.possible_vehicles(state)

    while container_deque:
        container = container_deque.popleft()
        variant = config.variants[container]

        if (state.pity_counter[container] >= variant.pity_threshold) or (np.random.random() < variant.vehicle_probability):
            if not possible_vehicles[container] > 0:
                container_deque.append("prime")
                state.received_containers["prime"] += 1
            else:
                if container not in state.received_vehicles:
                    state.received_vehicles[container] = 0

                state.received_vehicles[container] += 1
                possible_vehicles[container] -= 1
        
            state.pity_counter[container] = 0
        else:
            state.pity_counter[container] += 1

        # Handle container drops
        if np.random.random() < config.variants[container].container_probability:
            probabilities = config.intervariant_probabilities[container]
            extra_container = str(np.random.choice(a=list(probabilities.keys()), p=list(probabilities.values())))
            container_deque.append(extra_container)
            state.received_containers[extra_container] += 1

        state.increment_container(container)
        # Break if you have reached target
        if target_fn(state):
            break

        # if deque is empty, add one more box
        if not container_deque:
            container_deque.append("proto")

    return state


def run_parallel(fn: Callable[..., SimulationState], n_experiments: int) -> list[SimulationState]:
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(fn, range(n_experiments))
    return results
  