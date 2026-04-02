from __future__ import annotations

import numpy as np

from .model import SingleCartPoleParams


def _mass_matrix(state: np.ndarray, params: SingleCartPoleParams) -> np.ndarray:
    _, theta, _, _ = state
    m0 = params.cart_mass
    m1 = params.pole_mass
    l = params.pole_length
    return np.array(
        [
            [m0 + m1, m1 * l * np.cos(theta)],
            [m1 * l * np.cos(theta), m1 * l**2],
        ],
        dtype=float,
    )


def _generalized_forces(state: np.ndarray, cart_force: float, params: SingleCartPoleParams) -> np.ndarray:
    _, theta, x_dot, theta_dot = state
    m1 = params.pole_mass
    l = params.pole_length
    g = params.gravity
    return np.array(
        [
            cart_force - params.cart_damping * x_dot + m1 * l * np.sin(theta) * theta_dot**2,
            -params.joint_damping * theta_dot + m1 * g * l * np.sin(theta),
        ],
        dtype=float,
    )


def cartpole_dynamics(
    _time: float,
    state: np.ndarray,
    cart_force: float,
    params: SingleCartPoleParams,
) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    clipped_cart_force = np.clip(cart_force, -params.force_limit, params.force_limit)
    accelerations = np.linalg.solve(
        _mass_matrix(state, params),
        _generalized_forces(state, clipped_cart_force, params),
    )
    return np.array([state[2], state[3], accelerations[0], accelerations[1]], dtype=float)
