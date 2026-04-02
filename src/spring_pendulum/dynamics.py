from __future__ import annotations

import numpy as np

from .model import SpringCartPoleParams


def _mass_matrix(state: np.ndarray, params: SpringCartPoleParams) -> np.ndarray:
    _, theta, radius, _, _, _ = state
    m0 = params.cart_mass
    m1 = params.bob_mass
    return np.array(
        [
            [m0 + m1, m1 * radius * np.cos(theta), m1 * np.sin(theta)],
            [m1 * radius * np.cos(theta), m1 * radius**2, 0.0],
            [m1 * np.sin(theta), 0.0, m1],
        ],
        dtype=float,
    )


def _generalized_forces(state: np.ndarray, cart_force: float, params: SpringCartPoleParams) -> np.ndarray:
    _, theta, radius, x_dot, theta_dot, radius_dot = state
    m1 = params.bob_mass
    g = params.gravity
    return np.array(
        [
            cart_force
            - params.cart_damping * x_dot
            + m1 * radius * np.sin(theta) * theta_dot**2
            - 2.0 * m1 * np.cos(theta) * radius_dot * theta_dot,
            -2.0 * m1 * radius * radius_dot * theta_dot + m1 * g * radius * np.sin(theta) - params.angular_damping * theta_dot,
            m1 * radius * theta_dot**2
            - params.spring_constant * (radius - params.rest_length)
            - m1 * g * np.cos(theta)
            - params.radial_damping * radius_dot,
        ],
        dtype=float,
    )


def spring_cartpole_dynamics(
    _time: float,
    state: np.ndarray,
    cart_force: float,
    params: SpringCartPoleParams,
) -> np.ndarray:
    state = np.asarray(state, dtype=float).copy()
    state[2] = float(np.clip(state[2], params.min_length, params.max_length))
    clipped_cart_force = float(np.clip(cart_force, -params.force_limit, params.force_limit))
    accelerations = np.linalg.solve(_mass_matrix(state, params), _generalized_forces(state, clipped_cart_force, params))
    return np.array([state[3], state[4], state[5], accelerations[0], accelerations[1], accelerations[2]], dtype=float)
