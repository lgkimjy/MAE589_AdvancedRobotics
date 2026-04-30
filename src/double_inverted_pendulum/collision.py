from __future__ import annotations

import numpy as np

from .environment import BoxObstacle, CircularObstacle, Obstacle
from .kinematics import cart_and_tip_positions
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


def link_segments(state: np.ndarray, params: CartPoleParams) -> list[tuple[np.ndarray, np.ndarray]]:
    cart, joint1, tip = cart_and_tip_positions(state, params)
    return [(cart, joint1), (joint1, tip)]


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    alpha = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    closest = start + alpha * segment
    return float(np.linalg.norm(point - closest))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _on_segment(a: np.ndarray, point: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> bool:
    return bool(
        min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, eps: float = 1e-9) -> bool:
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)

    proper_cross = (
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    )
    if proper_cross:
        return True
    if abs(o1) <= eps and _on_segment(a0, b0, a1, eps):
        return True
    if abs(o2) <= eps and _on_segment(a0, b1, a1, eps):
        return True
    if abs(o3) <= eps and _on_segment(b0, a0, b1, eps):
        return True
    if abs(o4) <= eps and _on_segment(b0, a1, b1, eps):
        return True
    return False


def segment_obstacle_clearance(start: np.ndarray, end: np.ndarray, obstacle: Obstacle) -> float:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    center = np.asarray(obstacle.center, dtype=float)

    if isinstance(obstacle, CircularObstacle):
        return point_segment_distance(center, start, end) - obstacle.radius

    if isinstance(obstacle, BoxObstacle):
        half = np.array([0.5 * obstacle.width, 0.5 * obstacle.height], dtype=float)
        lower = center - half
        upper = center + half
        corners = [
            np.array([lower[0], lower[1]], dtype=float),
            np.array([upper[0], lower[1]], dtype=float),
            np.array([upper[0], upper[1]], dtype=float),
            np.array([lower[0], upper[1]], dtype=float),
        ]
        edges = list(zip(corners, corners[1:] + corners[:1]))

        start_inside = bool(np.all(start >= lower) and np.all(start <= upper))
        end_inside = bool(np.all(end >= lower) and np.all(end <= upper))
        if start_inside or end_inside or any(_segments_intersect(start, end, edge0, edge1) for edge0, edge1 in edges):
            return -min(obstacle.width, obstacle.height) * 0.5

        endpoint_clearance = min(point_obstacle_clearance(start, obstacle), point_obstacle_clearance(end, obstacle))
        corner_clearance = min(point_segment_distance(corner, start, end) for corner in corners)
        return float(min(endpoint_clearance, corner_clearance))

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
        for start, end in link_segments(state, params):
            for obstacle in obstacles:
                minimum = min(minimum, segment_obstacle_clearance(start, end, obstacle))
    return float(minimum)
