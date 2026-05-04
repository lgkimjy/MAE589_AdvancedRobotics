from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Rectangle

from double_inverted_pendulum.environment import BoxObstacle, CircularObstacle, Obstacle, goal_under_obstacle
from double_inverted_pendulum.model import default_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the obstacle penalty used in trajectory optimization.")
    parser.add_argument("--data", type=Path, default=Path("outputs/run3/data.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/obstacle_avoidance_explanation"))
    parser.add_argument("--clearance", type=float, default=0.24)
    parser.add_argument("--weight", type=float, default=1800.0)
    parser.add_argument("--samples-per-link", type=int, default=4)
    parser.add_argument(
        "--display-clearance",
        type=float,
        default=None,
        help="Clearance used only for drawing the dashed field. Defaults to --clearance.",
    )
    parser.add_argument(
        "--field-shape",
        choices=("ellipse", "box"),
        default="ellipse",
        help="Shape used to visualize and score the inflated obstacle field.",
    )
    parser.add_argument(
        "--show-distance-lines",
        action="store_true",
        help="Draw shortest Euclidean connector lines from sampled points to the nearest actual obstacle.",
    )
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def sample_points(state: np.ndarray, samples_per_link: int) -> tuple[np.ndarray, np.ndarray]:
    params = default_params()
    x_pos, theta1, theta2 = state[:3]
    cart = np.array([x_pos, 0.0], dtype=float)
    joint1 = cart + np.array(
        [params.link1_length * np.sin(theta1), params.link1_length * np.cos(theta1)],
        dtype=float,
    )
    tip = joint1 + np.array(
        [params.link2_length * np.sin(theta2), params.link2_length * np.cos(theta2)],
        dtype=float,
    )

    count = max(1, int(samples_per_link))
    points = [cart]
    for alpha in np.linspace(1.0 / count, 1.0, count):
        points.append(cart + alpha * (joint1 - cart))
    for alpha in np.linspace(1.0 / count, 1.0, count):
        points.append(joint1 + alpha * (tip - joint1))
    return np.asarray(points, dtype=float), np.vstack([cart, joint1, tip])


def intrusion_value(point: np.ndarray, obstacle, clearance: float, field_shape: str) -> float:
    point = np.asarray(point, dtype=float)
    center = np.asarray(obstacle.center, dtype=float)
    delta = point - center
    if isinstance(obstacle, CircularObstacle):
        safe_radius = max(float(obstacle.radius + clearance), 1.0e-9)
        return float(1.0 - np.dot(delta, delta) / safe_radius**2)
    if isinstance(obstacle, BoxObstacle):
        safe_half_width = max(float(0.5 * obstacle.width + clearance), 1.0e-9)
        safe_half_height = max(float(0.5 * obstacle.height + clearance), 1.0e-9)
        if field_shape == "box":
            normalized_distance = max(abs(delta[0]) / safe_half_width, abs(delta[1]) / safe_half_height)
            return float(1.0 - normalized_distance)
        return float(1.0 - (delta[0] / safe_half_width) ** 2 - (delta[1] / safe_half_height) ** 2)
    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")


def softplus(value: np.ndarray, beta: float = 12.0) -> np.ndarray:
    return np.logaddexp(0.0, beta * value) / beta


def closest_point_and_distance(point: np.ndarray, obstacle) -> tuple[np.ndarray, float]:
    point = np.asarray(point, dtype=float)
    center = np.asarray(obstacle.center, dtype=float)
    if isinstance(obstacle, CircularObstacle):
        delta = point - center
        norm = float(np.linalg.norm(delta))
        if norm <= 1.0e-12:
            closest = center + np.array([float(obstacle.radius), 0.0], dtype=float)
        else:
            closest = center + float(obstacle.radius) * delta / norm
        return closest, norm - float(obstacle.radius)

    if isinstance(obstacle, BoxObstacle):
        half = np.array([0.5 * obstacle.width, 0.5 * obstacle.height], dtype=float)
        lower = center - half
        upper = center + half
        closest = np.clip(point, lower, upper)
        outside_distance = float(np.linalg.norm(point - closest))
        if outside_distance > 1.0e-12:
            return closest, outside_distance

        distances_to_edges = np.array(
            [
                abs(point[0] - lower[0]),
                abs(upper[0] - point[0]),
                abs(point[1] - lower[1]),
                abs(upper[1] - point[1]),
            ],
            dtype=float,
        )
        edge = int(np.argmin(distances_to_edges))
        closest = point.copy()
        if edge == 0:
            closest[0] = lower[0]
        elif edge == 1:
            closest[0] = upper[0]
        elif edge == 2:
            closest[1] = lower[1]
        else:
            closest[1] = upper[1]
        return closest, -float(distances_to_edges[edge])

    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")


def nearest_obstacle_connectors(points: np.ndarray, obstacles: list) -> tuple[np.ndarray, np.ndarray]:
    if not obstacles:
        return points.copy(), np.full(points.shape[0], np.inf, dtype=float)

    closest_points = np.zeros_like(points)
    distances = np.full(points.shape[0], np.inf, dtype=float)
    for idx, point in enumerate(points):
        for obstacle in obstacles:
            closest, distance = closest_point_and_distance(point, obstacle)
            if abs(distance) < abs(distances[idx]):
                closest_points[idx] = closest
                distances[idx] = distance
    return closest_points, distances


def obstacle_metrics(
    states: np.ndarray,
    obstacles: list,
    clearance: float,
    weight: float,
    samples_per_link: int,
    field_shape: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    frame_cost = np.zeros(states.shape[0], dtype=float)
    frame_max_rho = np.full(states.shape[0], -np.inf, dtype=float)
    all_rho: list[np.ndarray] = []
    for frame, state in enumerate(states):
        points, _ = sample_points(state, samples_per_link)
        if not obstacles:
            all_rho.append(np.full(points.shape[0], -np.inf, dtype=float))
            frame_max_rho[frame] = -np.inf
            continue

        point_rho = []
        for point in points:
            values = np.array(
                [intrusion_value(point, obstacle, clearance, field_shape) for obstacle in obstacles],
                dtype=float,
            )
            point_rho.append(float(np.max(values)))
            frame_cost[frame] += float(weight) * float(np.sum(softplus(values) ** 2))
        all_rho.append(np.asarray(point_rho, dtype=float))
        frame_max_rho[frame] = float(np.max(point_rho))
    return frame_cost, frame_max_rho, all_rho


def draw_obstacles(axis, obstacles: list, clearance: float, field_shape: str, display_clearance: float) -> None:
    _ = clearance
    for obstacle in obstacles:
        if isinstance(obstacle, BoxObstacle):
            cx, cy = obstacle.center
            axis.add_patch(
                Rectangle(
                    (cx - 0.5 * obstacle.width, cy - 0.5 * obstacle.height),
                    obstacle.width,
                    obstacle.height,
                    facecolor=obstacle.color,
                    edgecolor=obstacle.color,
                    alpha=0.25,
                    linewidth=1.5,
                )
            )
            if field_shape == "box":
                axis.add_patch(
                    Rectangle(
                        (
                            cx - 0.5 * obstacle.width - display_clearance,
                            cy - 0.5 * obstacle.height - display_clearance,
                        ),
                        obstacle.width + 2.0 * display_clearance,
                        obstacle.height + 2.0 * display_clearance,
                        fill=False,
                        edgecolor=obstacle.color,
                        linestyle="--",
                        linewidth=1.6,
                        alpha=0.8,
                    )
                )
            else:
                axis.add_patch(
                    Ellipse(
                        (cx, cy),
                        width=2.0 * (0.5 * obstacle.width + display_clearance),
                        height=2.0 * (0.5 * obstacle.height + display_clearance),
                        fill=False,
                        edgecolor=obstacle.color,
                        linestyle="--",
                        linewidth=1.6,
                        alpha=0.8,
                    )
                )
        elif isinstance(obstacle, CircularObstacle):
            cx, cy = obstacle.center
            axis.add_patch(
                Ellipse(
                    (cx, cy),
                    width=2.0 * obstacle.radius,
                    height=2.0 * obstacle.radius,
                    facecolor=obstacle.color,
                    edgecolor=obstacle.color,
                    alpha=0.25,
                    linewidth=1.5,
                )
            )
            axis.add_patch(
                Ellipse(
                    (cx, cy),
                    width=2.0 * (obstacle.radius + display_clearance),
                    height=2.0 * (obstacle.radius + display_clearance),
                    fill=False,
                    edgecolor=obstacle.color,
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.8,
                )
            )


def point_colors(rho: np.ndarray) -> list[str]:
    colors = []
    for value in rho:
        if value > 0.0:
            colors.append("#d62828")
        elif value > -0.35:
            colors.append("#f77f00")
        else:
            colors.append("#1f77b4")
    return colors


def build_obstacles() -> list[Obstacle]:
    return list(goal_under_obstacle())


def animation_frame_indices(time: np.ndarray, fps: int) -> np.ndarray:
    if time.size == 0:
        raise ValueError("Cannot animate an empty time series.")

    frame_dt = 1.0 / fps
    frame_times = np.arange(time[0], time[-1] + 0.5 * frame_dt, frame_dt)
    frame_indices = np.searchsorted(time, frame_times, side="left")
    frame_indices = np.clip(frame_indices, 0, time.size - 1)
    return np.unique(frame_indices)


def make_figure(
    plan_time: np.ndarray,
    plan_state: np.ndarray,
    obstacles: list[Obstacle],
    clearance: float,
    weight: float,
    samples_per_link: int,
    output_dir: Path,
    fps: int,
    field_shape: str,
    display_clearance: float | None,
    show_distance_lines: bool,
    track_bounds: tuple[float, float] = (-1.5, 5.5),
    goal_x: float = 3.0,
    output_basename: str = "obstacle_penalty_geometry",
    save_static: bool = True,
    save_mp4: bool = True,
    append_style_suffix: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    display_clearance = clearance if display_clearance is None else display_clearance
    cost, max_rho, all_rho = obstacle_metrics(plan_state, obstacles, clearance, weight, samples_per_link, field_shape)
    params = default_params()
    if field_shape == "ellipse":
        suffix = ""
    elif abs(display_clearance) <= 1.0e-12:
        suffix = "_box_actual_distance" if show_distance_lines else "_box_actual"
    else:
        suffix = "_box"
    basename = f"{output_basename}{suffix if append_style_suffix else ''}"

    tip_path = []
    for state in plan_state:
        _, links = sample_points(state, samples_per_link)
        tip_path.append(links[-1])
    tip_path = np.asarray(tip_path, dtype=float)
    frame_indices = animation_frame_indices(plan_time, fps)
    finite_rho = np.where(np.isfinite(max_rho), max_rho, -np.inf)
    focus_frame = int(np.argmax(finite_rho))

    fig = plt.figure(figsize=(12.572, 4.8), dpi=140)
    ax = fig.add_axes((0.064, 0.1875, 0.5665, 0.5715))
    ax_cost = fig.add_axes((0.710, 0.1875, 0.246, 0.5715))

    def draw_frame(frame: int) -> None:
        ax.clear()
        ax_cost.clear()
        state = plan_state[frame]
        points, links = sample_points(state, samples_per_link)
        rho = all_rho[frame]
        actual_size_display = field_shape == "box" and abs(display_clearance) <= 1.0e-12
        closest_points, euclidean_distances = nearest_obstacle_connectors(points, obstacles)
        closest_idx = int(np.argmin(np.abs(euclidean_distances)))
        max_rho_text = "n/a" if not np.isfinite(max_rho[frame]) else f"{max_rho[frame]:.3f}"

        draw_obstacles(ax, obstacles, clearance, field_shape, display_clearance)
        ax.plot(tip_path[:, 0], tip_path[:, 1], color="#f2c94c", alpha=0.35, linewidth=2.0, label="tip path")
        if show_distance_lines and obstacles:
            for point_idx, (point, closest) in enumerate(zip(points, closest_points)):
                line_color = "#d62828" if point_idx == closest_idx else "#7f8c8d"
                line_width = 2.2 if point_idx == closest_idx else 1.0
                line_alpha = 0.95 if point_idx == closest_idx else 0.45
                ax.plot(
                    [point[0], closest[0]],
                    [point[1], closest[1]],
                    color=line_color,
                    linewidth=line_width,
                    alpha=line_alpha,
                    zorder=3,
                )
        ax.plot(links[:, 0], links[:, 1], "-o", color="#4b3f72", linewidth=4.0, markersize=6.8)
        ax.scatter(points[:, 0], points[:, 1], s=84, c=point_colors(rho), edgecolors="white", linewidths=0.95, zorder=5)
        if show_distance_lines and obstacles:
            ax.scatter(
                closest_points[:, 0],
                closest_points[:, 1],
                s=26,
                c="#d62828",
                marker="x",
                linewidths=1.2,
                alpha=0.8,
                zorder=6,
            )
        ax.scatter([links[0, 0]], [links[0, 1]], s=160, marker="s", c="#2f4050", edgecolors="black", zorder=4)
        ax.axhline(0.0, color="#777777", linewidth=1.1)
        ax.axvline(goal_x, ymin=0.53, ymax=0.88, color="#2d9cdb", linestyle="--", linewidth=1.6)
        ax.text(goal_x + 0.04, 1.06, "goal", color="#2d9cdb", fontsize=10)
        ax.text(
            0.02,
            0.96,
            f"t = {plan_time[frame]:.2f} s\n"
            f"sample points = {points.shape[0]}\n"
            f"max rho = {max_rho_text}\n"
            f"J_obs(k) = {cost[frame]:.2f}"
            + (f"\nmin d = {euclidean_distances[closest_idx]:.3f} m" if show_distance_lines and obstacles else ""),
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        ax.text(
            0.02,
            0.05,
            (
                "red/orange: sampled point inside or near obstacle penalty margin\n"
                "blue: sampled point outside obstacle penalty margin"
                if actual_size_display and show_distance_lines
                else "red/orange: sampled point inside or near actual obstacle\n"
                "blue: sampled point outside actual obstacle"
                if actual_size_display
                else "red/orange: sampled point inside or near inflated field\n"
                "blue: sampled point outside obstacle penalty field"
            ),
            transform=ax.transAxes,
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )
        if not obstacles:
            ax.set_title("Kinematic samples without active obstacle penalty")
        elif actual_size_display:
            title = "Kinematic samples with actual box obstacle geometry"
            if show_distance_lines:
                title = "Euclidean connectors from sampled points to actual obstacles"
            ax.set_title(title)
        else:
            title_shape = "box" if field_shape == "box" else "elliptic"
            ax.set_title(f"Kinematic samples with inflated {title_shape} obstacle field")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_xlim(*track_bounds)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)

        ax_cost.plot(plan_time, cost, color="#4b3f72", linewidth=2.0)
        ax_cost.axvline(plan_time[frame], color="#d62828", linewidth=1.6)
        ax_cost.fill_between(plan_time, 0.0, cost, color="#4b3f72", alpha=0.18)
        ax_cost.text(
            0.03,
            0.96,
            (
                "rho_o(p) > 0 means the\nsample point is inside the\nobstacle penalty margin."
                if actual_size_display and show_distance_lines
                else "rho_o(p) > 0 means the\nsample point is inside the\nactual box obstacle."
                if actual_size_display
                else "rho_o(p) > 0 means the\nsample point is inside the\ninflated obstacle field."
            ),
            transform=ax_cost.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        ax_cost.set_title("Instantaneous obstacle penalty")
        ax_cost.set_xlabel("time [s]")
        ax_cost.set_ylabel("J_obs(k)", labelpad=2)
        upper_cost = max(1.0, float(np.max(cost)))
        if upper_cost > 1.0:
            upper_cost = 50.0 * float(np.ceil(upper_cost / 50.0))
        ax_cost.set_ylim(0.0, upper_cost)
        ax_cost.grid(True, alpha=0.25)

    draw_frame(focus_frame)
    if save_static:
        fig.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
        fig.savefig(output_dir / f"{basename}.svg", bbox_inches="tight")

    def update(frame: int):
        draw_frame(frame)
        return []

    if save_mp4:
        animation = FuncAnimation(fig, update, frames=frame_indices, interval=1000.0 / fps, blit=False)
        animation.save(output_dir / f"{basename}.mp4", fps=fps, dpi=140)
    plt.close(fig)


def save_obstacle_explanation_video(
    time: np.ndarray,
    state: np.ndarray,
    obstacles: list[Obstacle],
    output_path: Path,
    clearance: float,
    weight: float,
    samples_per_link: int,
    fps: int,
    track_bounds: tuple[float, float],
    goal_x: float,
) -> None:
    make_figure(
        plan_time=np.asarray(time, dtype=float),
        plan_state=np.asarray(state, dtype=float),
        obstacles=obstacles,
        clearance=clearance,
        weight=weight,
        samples_per_link=samples_per_link,
        output_dir=output_path.parent,
        fps=fps,
        field_shape="box",
        display_clearance=0.0,
        show_distance_lines=True,
        track_bounds=(-1.5, 5.5),
        goal_x=goal_x,
        output_basename=output_path.stem,
        save_static=False,
        save_mp4=True,
        append_style_suffix=False,
    )


def main() -> None:
    args = parse_args()
    data = np.load(args.data)
    time_key = "simulation_time" if "simulation_time" in data else "plan_time"
    state_key = "simulation_state" if "simulation_state" in data else "plan_state"
    plan_time = np.asarray(data[time_key], dtype=float)
    plan_state = np.asarray(data[state_key], dtype=float)
    make_figure(
        plan_time=plan_time,
        plan_state=plan_state,
        obstacles=build_obstacles(),
        clearance=args.clearance,
        weight=args.weight,
        samples_per_link=args.samples_per_link,
        output_dir=args.output_dir,
        fps=args.fps,
        field_shape=args.field_shape,
        display_clearance=args.display_clearance,
        show_distance_lines=args.show_distance_lines,
    )
    print(f"Saved explanation visuals to {args.output_dir}")


if __name__ == "__main__":
    main()
