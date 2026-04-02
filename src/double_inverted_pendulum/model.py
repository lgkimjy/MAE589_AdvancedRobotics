from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartPoleParams:
    cart_mass: float = 1.0
    link1_mass: float = 0.2
    link2_mass: float = 0.2
    link1_length: float = 0.5
    link2_length: float = 0.5
    gravity: float = 9.81
    cart_damping: float = 0.1
    joint1_damping: float = 0.02
    joint2_damping: float = 0.02
    force_limit: float = 30.0


def default_params() -> CartPoleParams:
    return CartPoleParams()


def upright_state() -> np.ndarray:
    return np.zeros(6, dtype=float)


def downright_state(angle_perturbation: float = 0.0) -> np.ndarray:
    return np.array([0.0, np.pi + angle_perturbation, np.pi - angle_perturbation, 0.0, 0.0, 0.0], dtype=float)
