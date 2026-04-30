from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .environment import BoxObstacle, CircularObstacle, Obstacle
from .kinematics import end_effector_path, cart_and_tip_positions
from .model import CartPoleParams
from .simulation import SimulationResult


@dataclass(frozen=True)
class AnimationOptions:
    show_tip_trace: bool = False
    show_prediction: bool = False
    fps: int = 30
    cart_width: float = 0.24
    cart_height: float = 0.12
    rod_width: float = 3.0
    figure_size: tuple[float, float] = (9.0, 4.5)
    follow_cart: bool = False
    camera_width: float = 4.0
    y_limits: tuple[float, float] = (-1.25, 1.45)
    track_limit: float | None = None
    track_bounds: tuple[float, float] | None = None
    goal_x: float | None = None
    trace_window: int = 80
    #: Optional reference trajectory (e.g. cubic smoothstep warm-start for trajopt): drawn as a static tip path.
    reference_state: np.ndarray | None = None


def animate_simulation(
    sim_result: SimulationResult,
    params: CartPoleParams,
    show: bool = True,
    save_path: str | None = None,
    options: AnimationOptions | None = None,
    obstacles: list[Obstacle] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Rectangle

    opts = AnimationOptions() if options is None else options
    obstacles = [] if obstacles is None else obstacles
    states = sim_result.state
    if sim_result.time.size == 0:
        raise ValueError("Simulation result is empty.")
    tip_path = end_effector_path(states, params)
    all_positions = np.array([cart_and_tip_positions(state, params) for state in states], dtype=float)
    frame_dt = 1.0 / opts.fps
    frame_times = np.arange(sim_result.time[0], sim_result.time[-1] + 0.5 * frame_dt, frame_dt)
    frame_indices = np.searchsorted(sim_result.time, frame_times, side="left")
    frame_indices = np.clip(frame_indices, 0, sim_result.time.size - 1)
    frame_indices = np.unique(frame_indices)
    predicted_positions = None
    if sim_result.predicted_state is not None:
        predicted_positions = np.full(
            (sim_result.predicted_state.shape[0], sim_result.predicted_state.shape[1], 3, 2),
            np.nan,
            dtype=float,
        )
        for i in range(sim_result.predicted_state.shape[0]):
            for j in range(sim_result.predicted_state.shape[1]):
                if np.isnan(sim_result.predicted_state[i, j]).any():
                    continue
                predicted_positions[i, j] = cart_and_tip_positions(sim_result.predicted_state[i, j], params)
    predicted_sample_positions = None
    if sim_result.predicted_samples is not None:
        predicted_sample_positions = np.full(
            (
                sim_result.predicted_samples.shape[0],
                sim_result.predicted_samples.shape[1],
                sim_result.predicted_samples.shape[2],
                3,
                2,
            ),
            np.nan,
            dtype=float,
        )
        for i in range(sim_result.predicted_samples.shape[0]):
            for j in range(sim_result.predicted_samples.shape[1]):
                for k in range(sim_result.predicted_samples.shape[2]):
                    if np.isnan(sim_result.predicted_samples[i, j, k]).any():
                        continue
                    predicted_sample_positions[i, j, k] = cart_and_tip_positions(sim_result.predicted_samples[i, j, k], params)

    fig, ax = plt.subplots(figsize=opts.figure_size)
    x_extent = max(opts.camera_width / 2.0, params.link1_length + params.link2_length + 0.8)
    cart_positions = states[:, 0]
    obstacle_x = []
    for obstacle in obstacles:
        if isinstance(obstacle, CircularObstacle):
            obstacle_x.extend([obstacle.center[0] - obstacle.radius, obstacle.center[0] + obstacle.radius])
        else:
            half_width = obstacle.width / 2.0
            obstacle_x.extend([obstacle.center[0] - half_width, obstacle.center[0] + half_width])
    goal_candidates = [opts.goal_x] if opts.goal_x is not None else []
    if opts.track_bounds is not None:
        world_x_min = opts.track_bounds[0] - 0.25
        world_x_max = opts.track_bounds[1] + 0.25
    elif opts.track_limit is not None:
        world_x_min = -opts.track_limit - 0.35
        world_x_max = opts.track_limit + 0.35
    else:
        world_x_min = min([float(cart_positions.min())] + obstacle_x + goal_candidates) - 0.8
        world_x_max = max([float(cart_positions.max())] + obstacle_x + goal_candidates) + 0.8
    initial_cart_x = states[0, 0]
    if opts.follow_cart:
        ax.set_xlim(initial_cart_x - x_extent, initial_cart_x + x_extent)
    else:
        ax.set_xlim(world_x_min, world_x_max)
    ax.set_ylim(*opts.y_limits)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Double Inverted Pendulum Cart-Pole")
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0, color="#444444", linewidth=1.5, alpha=0.8)

    if opts.track_bounds is not None or opts.track_limit is not None:
        left_wall_x, right_wall_x = (
            opts.track_bounds if opts.track_bounds is not None else (-opts.track_limit, opts.track_limit)
        )
        wall_y0, wall_y1 = opts.y_limits[0] + 0.12, 0.28
        for wall_x, direction in ((left_wall_x, 1.0), (right_wall_x, -1.0)):
            ax.plot([wall_x, wall_x], [wall_y0, wall_y1], color="#8d99ae", linewidth=3.0)
            hatch_offsets = np.arange(wall_y0 + 0.015, wall_y1 - 0.01, 0.11)
            for offset in hatch_offsets:
                ax.plot(
                    [wall_x, wall_x - 0.12 * direction],
                    [offset, offset + 0.08],
                    color="#8d99ae",
                    linewidth=1.4,
                    alpha=0.9,
                )

    for obstacle in obstacles:
        if isinstance(obstacle, CircularObstacle):
            ax.add_patch(Circle(obstacle.center, obstacle.radius, color=obstacle.color, alpha=0.35))
            continue

        ax.add_patch(
            Rectangle(
                (obstacle.center[0] - obstacle.width / 2.0, obstacle.center[1] - obstacle.height / 2.0),
                obstacle.width,
                obstacle.height,
                facecolor=obstacle.color,
                edgecolor=obstacle.color,
                linewidth=1.2,
                alpha=0.35,
            )
        )

    cart_patch = Rectangle(
        (states[0, 0] - opts.cart_width / 2.0, -opts.cart_height / 2.0),
        opts.cart_width,
        opts.cart_height,
        facecolor="#355070",
        edgecolor="#1b263b",
        linewidth=1.5,
    )
    ax.add_patch(cart_patch)

    rod1_line, = ax.plot([], [], color="#6a4c93", linewidth=opts.rod_width)
    rod2_line, = ax.plot([], [], color="#6a4c93", linewidth=opts.rod_width)
    joint1_marker, = ax.plot([], [], "o", color="#1982c4", markersize=8)
    tip_marker, = ax.plot([], [], "o", color="#ffca3a", markersize=9)
    tip_trace_collection = LineCollection([], linewidths=2.0, zorder=2)
    ax.add_collection(tip_trace_collection)
    prediction_tip_line, = ax.plot([], [], color="#ff595e", linewidth=2.2, alpha=0.9, linestyle="--")
    prediction_cart_line, = ax.plot([], [], color="#1982c4", linewidth=1.8, alpha=0.55, linestyle=":")
    prediction_sample_collection = LineCollection([], colors=[], linewidths=1.2, zorder=1)
    ax.add_collection(prediction_sample_collection)
    text_box = dict(facecolor="white", alpha=0.88, edgecolor="none", boxstyle="round,pad=0.25")
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", bbox=text_box)
    force_text = ax.text(0.02, 0.88, "", transform=ax.transAxes, va="top", bbox=text_box)
    position_text = ax.text(0.02, 0.80, "", transform=ax.transAxes, va="top", bbox=text_box)
    goal_text = ax.text(0.02, 0.72, "", transform=ax.transAxes, va="top", bbox=text_box)

    if opts.goal_x is not None:
        goal_height = params.link1_length + params.link2_length
        ax.plot(
            [opts.goal_x, opts.goal_x],
            [0.0, goal_height],
            color="#1982c4",
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
        )
        ax.text(opts.goal_x, goal_height + 0.075, "goal", color="#1982c4", ha="center", va="bottom")

    if opts.reference_state is not None:
        ref_states = np.asarray(opts.reference_state, dtype=float)
        ref_tip_path = end_effector_path(ref_states, params)
        if ref_tip_path.shape[0] >= 2:
            ax.plot(
                ref_tip_path[:, 0],
                ref_tip_path[:, 1],
                color="#2a9d8f",
                linewidth=2.0,
                linestyle="--",
                alpha=0.9,
                zorder=1.85,
                label="smoothstep ref",
            )
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    def init() -> tuple[object, ...]:
        rod1_line.set_data([], [])
        rod2_line.set_data([], [])
        joint1_marker.set_data([], [])
        tip_marker.set_data([], [])
        tip_trace_collection.set_segments([])
        prediction_tip_line.set_data([], [])
        prediction_cart_line.set_data([], [])
        prediction_sample_collection.set_segments([])
        time_text.set_text("")
        force_text.set_text("")
        position_text.set_text("")
        goal_text.set_text("")
        return (
            cart_patch,
            rod1_line,
            rod2_line,
            joint1_marker,
            tip_marker,
            tip_trace_collection,
            prediction_tip_line,
            prediction_cart_line,
            prediction_sample_collection,
            time_text,
            force_text,
            position_text,
            goal_text,
        )

    def update(display_idx: int) -> tuple[object, ...]:
        frame_idx = int(frame_indices[display_idx])
        visible_indices = frame_indices[: display_idx + 1]
        positions = all_positions[frame_idx]
        cart, joint1_tip, joint2_tip = positions
        if opts.follow_cart:
            ax.set_xlim(cart[0] - x_extent, cart[0] + x_extent)
        cart_patch.set_xy((cart[0] - opts.cart_width / 2.0, -opts.cart_height / 2.0))
        rod1_line.set_data([cart[0], joint1_tip[0]], [cart[1], joint1_tip[1]])
        rod2_line.set_data([joint1_tip[0], joint2_tip[0]], [joint1_tip[1], joint2_tip[1]])
        joint1_marker.set_data([joint1_tip[0]], [joint1_tip[1]])
        tip_marker.set_data([joint2_tip[0]], [joint2_tip[1]])
        if opts.show_tip_trace:
            trace_start_display_idx = max(0, display_idx - opts.trace_window + 1)
            trace = tip_path[visible_indices[trace_start_display_idx:]]
            if trace.shape[0] >= 2:
                segments = np.stack([trace[:-1], trace[1:]], axis=1)
                alpha = np.linspace(0.06, 0.95, segments.shape[0]) ** 1.15
                colors = np.column_stack(
                    [
                        np.full(segments.shape[0], 1.0),
                        np.full(segments.shape[0], 202.0 / 255.0),
                        np.full(segments.shape[0], 58.0 / 255.0),
                        alpha,
                    ]
                )
                tip_trace_collection.set_segments(segments)
                tip_trace_collection.set_colors(colors)
            else:
                tip_trace_collection.set_segments([])
        else:
            tip_trace_collection.set_segments([])
        if opts.show_prediction and predicted_positions is not None:
            pred = predicted_positions[frame_idx]
            valid = ~np.isnan(pred[:, 0, 0])
            prediction_cart_line.set_data(pred[valid, 0, 0], pred[valid, 0, 1])
            prediction_tip_line.set_data(pred[valid, 2, 0], pred[valid, 2, 1])
            if predicted_sample_positions is not None:
                sample_pred = predicted_sample_positions[frame_idx]
                segments = []
                colors = []
                for sample_idx in range(sample_pred.shape[0]):
                    valid_sample = ~np.isnan(sample_pred[sample_idx, :, 2, 0])
                    path = sample_pred[sample_idx, valid_sample, 2, :]
                    if path.shape[0] >= 2:
                        segments.append(path)
                        colors.append((1.0, 0.35, 0.37, 0.08))
                prediction_sample_collection.set_segments(segments)
                prediction_sample_collection.set_colors(colors)
            else:
                prediction_sample_collection.set_segments([])
        else:
            prediction_cart_line.set_data([], [])
            prediction_tip_line.set_data([], [])
            prediction_sample_collection.set_segments([])
        time_text.set_text(f"t = {sim_result.time[frame_idx]:.2f} s")
        force_text.set_text(f"cart force = {sim_result.control[frame_idx]:.2f} N")
        position_text.set_text(f"cart x = {cart[0]:.2f} m")
        goal_text.set_text("" if opts.goal_x is None else f"goal x = {opts.goal_x:.2f} m")
        return (
            cart_patch,
            rod1_line,
            rod2_line,
            joint1_marker,
            tip_marker,
            tip_trace_collection,
            prediction_tip_line,
            prediction_cart_line,
            prediction_sample_collection,
            time_text,
            force_text,
            position_text,
            goal_text,
        )

    interval_ms = 1000.0 / opts.fps
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices.size,
        init_func=init,
        interval=interval_ms,
        blit=False,
    )

    fig.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            anim.save(output_path, writer="pillow", fps=opts.fps)
        elif suffix == ".mp4":
            anim.save(output_path, writer="ffmpeg", fps=opts.fps)
        else:
            raise ValueError(f"Unsupported animation format: {suffix}")

    if show:
        plt.show()
    else:
        plt.close(fig)
