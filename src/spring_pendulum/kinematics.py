from __future__ import annotations

import numpy as np

from .model import SpringCartPoleParams


def cart_and_tip_positions(state: np.ndarray, params: SpringCartPoleParams) -> np.ndarray:
    x_cart, theta, radius, _, _, _ = np.asarray(state, dtype=float)
    radius = float(np.clip(radius, params.min_length, params.max_length))
    cart = np.array([x_cart, 0.0], dtype=float)
    tip = cart + np.array([radius * np.sin(theta), radius * np.cos(theta)], dtype=float)
    return np.vstack([cart, tip])


def end_effector_path(state_history: np.ndarray, params: SpringCartPoleParams) -> np.ndarray:
    return np.array([cart_and_tip_positions(state, params)[-1] for state in np.asarray(state_history)], dtype=float)


def horizontal_span(state: np.ndarray, params: SpringCartPoleParams) -> tuple[float, float]:
    positions = cart_and_tip_positions(state, params)
    return float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
