from __future__ import annotations

import numpy as np

from .model import CartPoleParams


def _mass_matrix(state: np.ndarray, params: CartPoleParams) -> np.ndarray:
    _, theta1, theta2, _, _, _ = state
    m0 = params.cart_mass
    m1 = params.link1_mass
    m2 = params.link2_mass
    l1 = params.link1_length
    l2 = params.link2_length

    return np.array(
        [
            [m0 + m1 + m2, (m1 + m2) * l1 * np.cos(theta1), m2 * l2 * np.cos(theta2)],
            [(m1 + m2) * l1 * np.cos(theta1), (m1 + m2) * l1**2, m2 * l1 * l2 * np.cos(theta1 - theta2)],
            [m2 * l2 * np.cos(theta2), m2 * l1 * l2 * np.cos(theta1 - theta2), m2 * l2**2],
        ],
        dtype=float,
    )


def _generalized_forces(state: np.ndarray, cart_force: float, params: CartPoleParams) -> np.ndarray:
    _, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
    m1 = params.link1_mass
    m2 = params.link2_mass
    l1 = params.link1_length
    l2 = params.link2_length
    g = params.gravity

    q = np.array(
        [
            cart_force
            - params.cart_damping * x_dot
            + (m1 + m2) * l1 * np.sin(theta1) * theta1_dot**2
            + m2 * l2 * np.sin(theta2) * theta2_dot**2,
            -params.joint1_damping * theta1_dot
            - m2 * l1 * l2 * np.sin(theta1 - theta2) * theta2_dot**2
            + (m1 + m2) * g * l1 * np.sin(theta1),
            -params.joint2_damping * theta2_dot
            + m2 * l1 * l2 * np.sin(theta1 - theta2) * theta1_dot**2
            + m2 * g * l2 * np.sin(theta2),
        ],
        dtype=float,
    )
    return q


def cartpole_dynamics(
    _time: float,
    state: np.ndarray,
    cart_force: float,
    params: CartPoleParams,
) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    clipped_cart_force = np.clip(cart_force, -params.force_limit, params.force_limit)
    accelerations = np.linalg.solve(
        _mass_matrix(state, params),
        _generalized_forces(state, clipped_cart_force, params),
    )
    return np.array(
        [
            state[3],
            state[4],
            state[5],
            accelerations[0],
            accelerations[1],
            accelerations[2],
        ],
        dtype=float,
    )


def linear_state_space(
    params: CartPoleParams,
    equilibrium_state: np.ndarray | None = None,
    equilibrium_input: float = 0.0,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    x_eq = np.zeros(6, dtype=float) if equilibrium_state is None else np.asarray(equilibrium_state, dtype=float)
    u_eq = float(equilibrium_input)
    n_states = x_eq.size
    a_mat = np.zeros((n_states, n_states), dtype=float)

    for idx in range(n_states):
        dx = np.zeros(n_states, dtype=float)
        dx[idx] = epsilon
        f_plus = cartpole_dynamics(0.0, x_eq + dx, u_eq, params)
        f_minus = cartpole_dynamics(0.0, x_eq - dx, u_eq, params)
        a_mat[:, idx] = (f_plus - f_minus) / (2.0 * epsilon)

    f_plus = cartpole_dynamics(0.0, x_eq, u_eq + epsilon, params)
    f_minus = cartpole_dynamics(0.0, x_eq, u_eq - epsilon, params)
    b_mat = ((f_plus - f_minus) / (2.0 * epsilon)).reshape(-1, 1)
    return a_mat, b_mat
