from __future__ import annotations

import numpy as np

from .environment import BoxObstacle, CircularObstacle, Obstacle
from .model import CartPoleParams


def link_sample_points(state: np.ndarray, params: CartPoleParams, samples_per_link: int = 4) -> np.ndarray:
    x_pos, theta1, theta2, _, _, _ = np.asarray(state, dtype=float)
    sample_count = max(1, int(samples_per_link))
    l1 = params.link1_length
    l2 = params.link2_length

    cart = np.array([x_pos, 0.0], dtype=float)
    joint1 = cart + np.array([l1 * np.sin(theta1), l1 * np.cos(theta1)], dtype=float)
    points = [cart]
    for alpha in np.linspace(1.0 / sample_count, 1.0, sample_count):
        points.append(cart + alpha * (joint1 - cart))
    for alpha in np.linspace(1.0 / sample_count, 1.0, sample_count):
        points.append(joint1 + alpha * np.array([l2 * np.sin(theta2), l2 * np.cos(theta2)], dtype=float))
    return np.asarray(points, dtype=float)


def point_obstacle_clearance(point: np.ndarray, obstacle: Obstacle) -> float:
    point = np.asarray(point, dtype=float)
    center = np.asarray(obstacle.center, dtype=float)
    delta = point - center

    if isinstance(obstacle, CircularObstacle):
        return float(np.linalg.norm(delta) - obstacle.radius)

    if isinstance(obstacle, BoxObstacle):
        outside = np.abs(delta) - np.array([0.5 * obstacle.width, 0.5 * obstacle.height], dtype=float)
        outside_distance = np.linalg.norm(np.maximum(outside, 0.0))
        inside_distance = min(max(float(outside[0]), float(outside[1])), 0.0)
        return float(outside_distance + inside_distance)

    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")


def minimum_obstacle_clearance(
    state_history: np.ndarray,
    params: CartPoleParams,
    obstacles: list[Obstacle],
    samples_per_link: int = 4,
) -> float:
    if not obstacles:
        return float("nan")

    minimum = float("inf")
    for state in np.asarray(state_history, dtype=float):
        for point in link_sample_points(state, params, samples_per_link=samples_per_link):
            for obstacle in obstacles:
                minimum = min(minimum, point_obstacle_clearance(point, obstacle))
    return float(minimum)
