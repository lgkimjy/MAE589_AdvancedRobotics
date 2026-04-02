from __future__ import annotations

import numpy as np

from .model import CartPoleParams


def cart_and_tip_positions(state: np.ndarray, params: CartPoleParams) -> np.ndarray:
    x_cart, theta1, theta2, _, _, _ = np.asarray(state, dtype=float)
    l1 = params.link1_length
    l2 = params.link2_length

    cart = np.array([x_cart, 0.0], dtype=float)
    joint1_tip = cart + np.array([l1 * np.sin(theta1), l1 * np.cos(theta1)], dtype=float)
    joint2_tip = joint1_tip + np.array([l2 * np.sin(theta2), l2 * np.cos(theta2)], dtype=float)
    return np.vstack([cart, joint1_tip, joint2_tip])


def end_effector_path(state_history: np.ndarray, params: CartPoleParams) -> np.ndarray:
    return np.array([cart_and_tip_positions(state, params)[-1] for state in np.asarray(state_history)], dtype=float)


def horizontal_span(state: np.ndarray, params: CartPoleParams) -> tuple[float, float]:
    positions = cart_and_tip_positions(state, params)
    return float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
