from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import SingleCartPoleParams, default_params, downright_state
from .simulation import ControlLaw, SimulationResult, rollout_open_loop


@dataclass
class SingleInvertedPendulumEnv:
    params: SingleCartPoleParams = field(default_factory=default_params)
    dt: float = 0.02
    t_final: float = 10.0
    cart_position_bounds: tuple[float, float] = (-1.0, 4.0)
    enforce_link_limits: bool = True
    initial_state: np.ndarray = field(default_factory=lambda: downright_state())

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.initial_state = np.asarray(state, dtype=float)
        return self.initial_state.copy()

    def rollout(self, controller: ControlLaw | None = None, initial_state: np.ndarray | None = None) -> SimulationResult:
        start = self.initial_state if initial_state is None else np.asarray(initial_state, dtype=float)
        control_law = controller if controller is not None else (lambda _t, _state: 0.0)
        return rollout_open_loop(
            initial_state=start,
            controller=control_law,
            params=self.params,
            t_final=self.t_final,
            dt=self.dt,
            position_bounds=self.cart_position_bounds,
            enforce_link_limits=self.enforce_link_limits,
        )
