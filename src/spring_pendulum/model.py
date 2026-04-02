from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpringCartPoleParams:
    cart_mass: float = 1.0
    bob_mass: float = 0.3
    rest_length: float = 0.7
    spring_constant: float = 50.0
    gravity: float = 9.81
    cart_damping: float = 0.08
    radial_damping: float = 0.12
    angular_damping: float = 0.035
    force_limit: float = 30.0
    min_length: float = 0.35
    max_length: float = 1.25


def default_params() -> SpringCartPoleParams:
    return SpringCartPoleParams()


def hanging_length(params: SpringCartPoleParams | None = None) -> float:
    use_params = default_params() if params is None else params
    return use_params.rest_length + use_params.bob_mass * use_params.gravity / use_params.spring_constant


def upright_state(params: SpringCartPoleParams | None = None) -> np.ndarray:
    use_params = default_params() if params is None else params
    return np.array([0.0, 0.0, use_params.rest_length, 0.0, 0.0, 0.0], dtype=float)


def downright_state(
    angle_perturbation: float = 0.0,
    length_offset: float = 0.0,
    params: SpringCartPoleParams | None = None,
) -> np.ndarray:
    use_params = default_params() if params is None else params
    return np.array(
        [0.0, np.pi + angle_perturbation, hanging_length(use_params) + length_offset, 0.0, 0.0, 0.0],
        dtype=float,
    )
