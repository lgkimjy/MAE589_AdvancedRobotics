from __future__ import annotations

import numpy as np

from .model import SingleCartPoleParams


def cart_and_tip_positions(state: np.ndarray, params: SingleCartPoleParams) -> np.ndarray:
    x_cart, theta, _, _ = np.asarray(state, dtype=float)
    cart = np.array([x_cart, 0.0], dtype=float)
    tip = cart + np.array(
        [params.pole_length * np.sin(theta), params.pole_length * np.cos(theta)],
        dtype=float,
    )
    return np.vstack([cart, tip])


def end_effector_path(state_history: np.ndarray, params: SingleCartPoleParams) -> np.ndarray:
    return np.array([cart_and_tip_positions(state, params)[-1] for state in np.asarray(state_history)], dtype=float)


def horizontal_span(state: np.ndarray, params: SingleCartPoleParams) -> tuple[float, float]:
    positions = cart_and_tip_positions(state, params)
    return float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
