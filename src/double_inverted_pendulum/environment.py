from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import CartPoleParams, default_params, downright_state
from .simulation import ControlLaw, SimulationResult, rollout_open_loop


@dataclass(frozen=True)
class CircularObstacle:
    center: tuple[float, float]
    radius: float
    color: str = "#b56576"


def default_track_obstacles() -> list[CircularObstacle]:
    return [
        CircularObstacle(center=(0.85, 0.05), radius=0.12, color="#d66853"),
        CircularObstacle(center=(1.15, 0.05), radius=0.12, color="#d66853"),
    ]


@dataclass
class DoubleInvertedPendulumEnv:
    params: CartPoleParams = field(default_factory=default_params)
    dt: float = 0.02
    t_final: float = 10.0
    cart_position_bounds: tuple[float, float] = (-1.0, 4.0)
    enforce_link_limits: bool = True
    initial_state: np.ndarray = field(default_factory=lambda: downright_state())
    obstacles: list[CircularObstacle] = field(default_factory=list)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.initial_state = np.asarray(state, dtype=float)
        return self.initial_state.copy()

    def rollout(self, controller: ControlLaw, initial_state: np.ndarray | None = None) -> SimulationResult:
        start = self.initial_state if initial_state is None else np.asarray(initial_state, dtype=float)
        return rollout_open_loop(
            initial_state=start,
            controller=controller,
            params=self.params,
            t_final=self.t_final,
            dt=self.dt,
            position_bounds=self.cart_position_bounds,
            enforce_link_limits=self.enforce_link_limits,
        )
