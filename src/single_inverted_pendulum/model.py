from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingleCartPoleParams:
    cart_mass: float = 1.0
    pole_mass: float = 0.25
    pole_length: float = 0.7
    gravity: float = 9.81
    cart_damping: float = 0.1
    joint_damping: float = 0.03
    force_limit: float = 30.0


def default_params() -> SingleCartPoleParams:
    return SingleCartPoleParams()


def upright_state() -> np.ndarray:
    return np.zeros(4, dtype=float)


def downright_state(angle_perturbation: float = 0.0) -> np.ndarray:
    return np.array([0.0, np.pi + angle_perturbation, 0.0, 0.0], dtype=float)
