from dataclasses import dataclass

from wot.montecarlo import SimulationConfig, VariantConfig, SimulationState

proto = VariantConfig(
    name="proto",
    vehicle_probability=0.02,
    possible_vehicles=10,
    container_probability=0.2,
    pity_threshold=50
)


alpha = VariantConfig(
    name="alpha",
    vehicle_probability=0.05,
    possible_vehicles=5,
    container_probability=0.2,
    pity_threshold=40
)


prime = VariantConfig(
    name="prime",
    vehicle_probability=0.1,
    possible_vehicles=3,
    container_probability=0.2,
    pity_threshold=20
)


pandora_config = SimulationConfig(
    variants={v.name: v for v in (proto, alpha, prime)},
    intervariant_probabilities={
        "proto": {"proto": 0.23, "alpha": 0.75, "prime": 0.02},
        "alpha": {"proto": 0.02, "alpha": 0.23, "prime": 0.75},
        "prime": {"proto": 0.75, "alpha": 0.23, "prime": 0.02},
    },
    preowned_vehicles={
        "proto": 0,
        "alpha": 0
    }
)


def all_prime_received(state: SimulationState) -> bool:
    return state.received_vehicles.get("prime", 0) >= 3


@dataclass(kw_only=True)
class NPurchasedTarget:
    n_proto: int = 0
    
    def __call__(self, state: SimulationState) -> bool:
        return (state.opened_containers.get("proto", 0) - state.received_containers.get("proto", 0) >= self.n_proto)
