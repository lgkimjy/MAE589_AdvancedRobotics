from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import site
import sys
import sysconfig
import tempfile
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def maybe_reexec_course_env() -> None:
    if os.environ.get("MAE589_TRAJOPT_NO_REEXEC"):
        return

    explicit_python = os.environ.get("MAE589_TRAJOPT_PYTHON")
    candidates = [Path(explicit_python)] if explicit_python else []
    candidates.extend(
        [
            Path.home() / "miniforge3/envs/mae589-advanced-robotics/bin/python",
            Path.home() / "miniconda3/envs/mae589-advanced-robotics/bin/python",
            Path.home() / "anaconda3/envs/mae589-advanced-robotics/bin/python",
        ]
    )

    current_python = Path(sys.executable).resolve()
    for candidate in candidates:
        if not candidate or not candidate.exists():
            continue
        candidate = candidate.resolve()
        if candidate == current_python:
            return
        os.environ["MAE589_TRAJOPT_REEXECED"] = "1"
        os.execv(str(candidate), [str(candidate), *sys.argv])


maybe_reexec_course_env()

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

for package_dir in [*site.getsitepackages(), sysconfig.get_paths().get("purelib", "")]:
    if not package_dir:
        continue
    if package_dir in sys.path:
        sys.path.remove(package_dir)
    sys.path.insert(1, package_dir)

from double_inverted_pendulum.collision import minimum_obstacle_clearance
from double_inverted_pendulum.controllers import TrajectorySwitchingController, design_equilibrium_lqr, state_error
from double_inverted_pendulum.environment import BoxObstacle, CircularObstacle, Obstacle, default_track_obstacles, goal_under_obstacle
from double_inverted_pendulum.kinematics import cart_and_tip_positions, end_effector_path
from double_inverted_pendulum.model import CartPoleParams, default_params, downright_state, upright_state
from double_inverted_pendulum.optimal_control import TrajectoryPlan, optimize_trajectory_with_casadi
from double_inverted_pendulum.simulation import SimulationResult, rollout_open_loop
from double_inverted_pendulum.visualization import AnimationOptions, animate_simulation
from scripts.visualize_obstacle_avoidance_meaning import save_obstacle_explanation_video


@dataclass(frozen=True)
class PaperRunConfig:
    name: str
    output_dir: Path
    track_bounds: tuple[float, float]
    obstacles: tuple[Obstacle, ...] = ()
    stabilize: bool = False
    horizon_steps: int = 80
    dt: float = 0.04
    hold_time: float = 6.0
    goal_x: float = 3.0
    initial_angle_offset: float = 0.08
    obstacle_weight: float = 1800.0
    obstacle_clearance: float = 0.24
    obstacle_samples_per_link: int = 4


@dataclass
class ModeRecorder:
    controller: TrajectorySwitchingController
    switch_time: float | None = None

    def __call__(self, time: float, state: np.ndarray) -> float:
        control = self.controller(time, state)
        if self.switch_time is None and self.controller.active_mode == "lqr":
            self.switch_time = float(time)
        return control

    @property
    def active_mode(self) -> str:
        return self.controller.active_mode


def run_case(config: PaperRunConfig) -> None:
    temp_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    params = replace(
        default_params(),
        cart_damping=1.0,
        joint1_damping=0.1,
        joint2_damping=0.01,
        force_limit=150.0,
    )
    initial_state = downright_state(config.initial_angle_offset)
    goal_state = upright_state()
    goal_state[0] = config.goal_x
    obstacles = list(config.obstacles)

    print(f"[{config.name}] solving trajectory optimization...")
    plan = optimize_trajectory_with_casadi(
        initial_state=initial_state,
        goal_state=goal_state,
        params=params,
        horizon_steps=config.horizon_steps,
        dt=config.dt,
        position_bounds=config.track_bounds,
        obstacles=obstacles,
        obstacle_clearance=config.obstacle_clearance,
        obstacle_weight=config.obstacle_weight,
        obstacle_samples_per_link=config.obstacle_samples_per_link,
    )

    switch_time = None
    if config.stabilize:
        final_lqr = design_equilibrium_lqr(params=params, equilibrium_state=goal_state)
        controller = ModeRecorder(
            TrajectorySwitchingController(
                params=params,
                plan_time=plan.time,
                plan_state=plan.state,
                plan_control=plan.control,
                equilibrium_state=goal_state,
                final_lqr=final_lqr,
            )
        )
        sim = rollout_open_loop(
            initial_state=initial_state,
            controller=controller,
            params=params,
            t_final=float(plan.time[-1] + max(config.hold_time, 0.0)),
            dt=config.dt,
            position_bounds=config.track_bounds,
            enforce_link_limits=True,
        )
        switch_time = controller.switch_time
    else:
        sim = SimulationResult(time=plan.time, state=plan.state, control=plan.control)

    save_data(config, params, plan, sim, obstacles, switch_time)
    cleanup_legacy_outputs(config.output_dir)
    save_plan_figure(config, params, goal_state, plan, sim, obstacles, switch_time)
    save_time_series_plan_figure(config, params, plan, sim, switch_time)
    animation_reference_state = plan.reference_state if not config.obstacles else None
    save_animations(config, params, goal_state, sim, obstacles, animation_reference_state)
    save_obstacle_explanation_video(
        time=sim.time,
        state=sim.state,
        obstacles=obstacles,
        output_path=config.output_dir / "obstacle_penalty_geometry.mp4",
        clearance=config.obstacle_clearance,
        weight=config.obstacle_weight,
        samples_per_link=config.obstacle_samples_per_link,
        fps=30,
        track_bounds=config.track_bounds,
        goal_x=goal_state[0],
    )
    print_summary(config, params, goal_state, plan, sim, obstacles, switch_time)


def cleanup_legacy_outputs(output_dir: Path) -> None:
    for filename in (
        "traj_plan2.pdf",
        "traj_plan2.svg",
        "traj_plan2.png",
        "trajopt_plan.png",
        "trajopt_plan2.png",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def save_data(
    config: PaperRunConfig,
    params: CartPoleParams,
    plan: TrajectoryPlan,
    sim: SimulationResult,
    obstacles: list[Obstacle],
    switch_time: float | None,
) -> None:
    np.savez(
        config.output_dir / "data.npz",
        plan_time=plan.time,
        plan_state=plan.state,
        plan_control=plan.control,
        reference_state=plan.reference_state,
        reference_control=plan.reference_control,
        simulation_time=sim.time,
        simulation_state=sim.state,
        simulation_control=sim.control,
        switch_time=np.nan if switch_time is None else switch_time,
        track_bounds=np.asarray(config.track_bounds, dtype=float),
    )

    plan_table = np.column_stack([plan.time, plan.control, plan.state])
    sim_table = np.column_stack([sim.time, sim.control, sim.state])
    header = "time,control,x,theta1,theta2,xdot,theta1dot,theta2dot"
    np.savetxt(config.output_dir / "plan.csv", plan_table, delimiter=",", header=header, comments="")
    np.savetxt(config.output_dir / "simulation.csv", sim_table, delimiter=",", header=header, comments="")

    metadata = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "params": asdict(params),
        "obstacles": [obstacle_to_dict(obstacle) for obstacle in obstacles],
        "switch_time": switch_time,
    }
    with (config.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def obstacle_to_dict(obstacle: Obstacle) -> dict[str, object]:
    if isinstance(obstacle, CircularObstacle):
        return {
            "type": "circle",
            "center": obstacle.center,
            "radius": obstacle.radius,
            "color": obstacle.color,
        }
    if isinstance(obstacle, BoxObstacle):
        return {
            "type": "box",
            "center": obstacle.center,
            "width": obstacle.width,
            "height": obstacle.height,
            "color": obstacle.color,
        }
    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")


def save_plan_figure(
    config: PaperRunConfig,
    params: CartPoleParams,
    goal_state: np.ndarray,
    plan: TrajectoryPlan,
    sim: SimulationResult,
    obstacles: list[Obstacle],
    switch_time: float | None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig = plt.figure(figsize=(10.5, 6.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.1, 1.0])
    ax_workspace = fig.add_subplot(grid[0, :])
    ax_x = fig.add_subplot(grid[1, 0])
    ax_u = fig.add_subplot(grid[1, 1])

    plan_tip = end_effector_path(plan.state, params)
    sim_tip = end_effector_path(sim.state, params)
    plan_cart = plan.state[:, 0]
    sim_cart = sim.state[:, 0]

    ax_workspace.axhline(0.0, color="#444444", linewidth=1.4, alpha=0.8)
    ax_workspace.plot(plan_cart, np.zeros_like(plan_cart), color="#355070", linewidth=1.8, alpha=0.75, label="cart path")
    ax_workspace.plot(plan_tip[:, 0], plan_tip[:, 1], color="#ffca3a", linewidth=2.5, alpha=0.95, label="planned tip path")
    if config.stabilize:
        ax_workspace.plot(sim_tip[:, 0], sim_tip[:, 1], color="#6a4c93", linewidth=1.3, alpha=0.55, label="executed tip path")

    draw_body(ax_workspace, plan.state[0], params, color="#555555", alpha=0.65)
    draw_body(ax_workspace, plan.state[-1], params, color="#2a9d8f", alpha=0.85)
    draw_obstacles(ax_workspace, obstacles)

    goal_height = params.link1_length + params.link2_length
    ax_workspace.plot([goal_state[0], goal_state[0]], [0.0, goal_height], color="#1982c4", linestyle="--", linewidth=1.8)
    ax_workspace.scatter([goal_state[0]], [goal_height], color="#1982c4", s=35, zorder=5, label="goal")

    if switch_time is not None:
        switch_state = interpolate_state(sim, switch_time)
        switch_positions = cart_and_tip_positions(switch_state, params)
        ax_workspace.scatter(
            [switch_positions[-1, 0]],
            [switch_positions[-1, 1]],
            marker="D",
            color="#d00000",
            s=45,
            zorder=6,
            label="LQR switch",
        )

    ax_workspace.set_xlim(config.track_bounds[0] - 0.25, config.track_bounds[1] + 0.25)
    ax_workspace.set_ylim(-1.25, 1.45)
    ax_workspace.set_aspect("equal")
    ax_workspace.set_xlabel("x [m]")
    ax_workspace.set_ylabel("y [m]")
    ax_workspace.grid(True, alpha=0.25)
    ax_workspace.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax_x.plot(plan.time, plan.state[:, 0], color="#355070", linewidth=2.0, label="planned cart x")
    if config.stabilize:
        ax_x.plot(sim.time, sim_cart, color="#6a4c93", linewidth=1.4, alpha=0.8, label="executed cart x")
    ax_x.axhline(goal_state[0], color="#1982c4", linestyle="--", linewidth=1.4, label="goal x")
    ax_x.set_xlabel("time [s]")
    ax_x.set_ylabel("cart x [m]")
    ax_x.grid(True, alpha=0.25)

    ax_u.plot(plan.time, plan.control, color="#d66853", linewidth=2.0, label="planned force")
    if config.stabilize:
        ax_u.plot(sim.time, sim.control, color="#8d0801", linewidth=1.2, alpha=0.75, label="executed force")
    ax_u.set_xlabel("time [s]")
    ax_u.set_ylabel("force [N]")
    ax_u.grid(True, alpha=0.25)

    if switch_time is not None:
        for axis in (ax_x, ax_u):
            axis.axvline(switch_time, color="#d00000", linestyle="--", linewidth=1.4, label="LQR switch")

    deduplicate_legend(ax_x)
    deduplicate_legend(ax_u)

    for suffix in ("pdf", "svg"):
        fig.savefig(config.output_dir / f"trajopt_plan.{suffix}", dpi=300)
    plt.close(fig)


def draw_obstacles(axis, obstacles: list[Obstacle]) -> None:
    from matplotlib.patches import Circle, Rectangle

    for obstacle in obstacles:
        if isinstance(obstacle, CircularObstacle):
            axis.add_patch(Circle(obstacle.center, obstacle.radius, color=obstacle.color, alpha=0.35))
            continue
        axis.add_patch(
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


def draw_body(axis, state: np.ndarray, params: CartPoleParams, color: str, alpha: float) -> None:
    positions = cart_and_tip_positions(state, params)
    axis.plot(positions[:2, 0], positions[:2, 1], color=color, linewidth=2.2, alpha=alpha)
    axis.plot(positions[1:, 0], positions[1:, 1], color=color, linewidth=2.2, alpha=alpha)
    axis.scatter(positions[:, 0], positions[:, 1], color=color, s=[24, 22, 28], alpha=alpha, zorder=5)


def interpolate_state(sim: SimulationResult, time: float) -> np.ndarray:
    state = np.empty(sim.state.shape[1], dtype=float)
    clamped = float(np.clip(time, sim.time[0], sim.time[-1]))
    for idx in range(sim.state.shape[1]):
        state[idx] = float(np.interp(clamped, sim.time, sim.state[:, idx]))
    return state


def deduplicate_legend(axis) -> None:
    handles, labels = axis.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axis.legend(unique.values(), unique.keys(), fontsize=8, framealpha=0.9)


def save_time_series_plan_figure(
    config: PaperRunConfig,
    params: CartPoleParams,
    plan: TrajectoryPlan,
    sim: SimulationResult,
    switch_time: float | None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 5.8), sharex=True, constrained_layout=True)
    time = plan.time
    show_reference = not config.obstacles

    suffix = "_plan" if config.stabilize else ""

    axes[0].plot(time, plan.state[:, 0], color="#1f77b4", linewidth=2.0, label=f"x{suffix}")
    if show_reference:
        axes[0].plot(time, plan.reference_state[:, 0], color="#1f77b4", linestyle="--", linewidth=1.4, label="x_ref")
    if config.stabilize:
        axes[0].plot(sim.time, sim.state[:, 0], color="#6a4c93", linewidth=1.3, alpha=0.75, label="x_exec")
    axes[0].set_ylabel("x [m]")

    axes[1].plot(time, plan.state[:, 1], color="#ff7f0e", linewidth=2.0, label=f"theta1{suffix}")
    axes[1].plot(time, plan.state[:, 2], color="#2ca02c", linewidth=2.0, label=f"theta2{suffix}")
    if show_reference:
        axes[1].plot(time, plan.reference_state[:, 1], color="#ff7f0e", linestyle="--", linewidth=1.4, label="theta1_ref")
        axes[1].plot(time, plan.reference_state[:, 2], color="#2ca02c", linestyle="--", linewidth=1.4, label="theta2_ref")
    if config.stabilize:
        axes[1].plot(sim.time, sim.state[:, 1], color="#ff7f0e", linewidth=1.1, alpha=0.55, label="theta1_exec")
        axes[1].plot(sim.time, sim.state[:, 2], color="#2ca02c", linewidth=1.1, alpha=0.55, label="theta2_exec")
    axes[1].set_ylabel("angle [rad]")

    axes[2].plot(time, plan.control, color="#d62728", linewidth=2.0, label=f"u{suffix}")
    if show_reference:
        reference_control_time = time[: plan.reference_control.shape[0]]
        axes[2].plot(
            reference_control_time,
            plan.reference_control,
            color="#d62728",
            linestyle="--",
            linewidth=1.4,
            label="u_ref",
        )
    if config.stabilize:
        axes[2].plot(sim.time, sim.control, color="#8d0801", linewidth=1.1, alpha=0.75, label="u_exec")
    axes[2].axhline(params.force_limit, color="black", linestyle="--", linewidth=1.2)
    axes[2].axhline(-params.force_limit, color="black", linestyle="--", linewidth=1.2)
    control_series = [plan.control]
    if config.stabilize:
        control_series.append(sim.control)
    stacked_control = np.concatenate(control_series)
    control_margin = max(5.0, 0.12 * float(np.ptp(stacked_control)))
    axes[2].set_ylim(float(np.min(stacked_control) - control_margin), float(np.max(stacked_control) + control_margin))
    axes[2].set_ylabel("u [N]")
    axes[2].set_xlabel("time [s]")

    if switch_time is not None:
        for axis in axes:
            axis.axvline(switch_time, color="#d00000", linestyle="-.", linewidth=1.4, label="LQR switch")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        deduplicate_legend(axis)

    for suffix in ("pdf", "svg"):
        fig.savefig(config.output_dir / f"trajopt_plan2.{suffix}", dpi=300)
    plt.close(fig)


def save_animations(
    config: PaperRunConfig,
    params: CartPoleParams,
    goal_state: np.ndarray,
    sim: SimulationResult,
    obstacles: list[Obstacle],
    reference_state: np.ndarray | None,
) -> None:
    base_options = dict(
        show_tip_trace=True,
        track_bounds=config.track_bounds,
        goal_x=goal_state[0],
        figure_size=(9.0, 4.5),
        dpi=300,
        fps=30,
    )
    animate_simulation(
        sim_result=sim,
        params=params,
        show=False,
        save_path=str(config.output_dir / "simulation.mp4"),
        options=AnimationOptions(**base_options, reference_state=reference_state, show_info_text=False),
        obstacles=obstacles,
    )
    animate_simulation(
        sim_result=sim,
        params=params,
        show=False,
        save_path=str(config.output_dir / "simulation2.mp4"),
        options=AnimationOptions(
            **base_options,
            reference_state=reference_state,
            show_info_text=False,
            trace_window=None,
            fade_tip_trace=False,
        ),
        obstacles=obstacles,
    )
    animate_simulation(
        sim_result=sim,
        params=params,
        show=False,
        save_path=str(config.output_dir / "simulation_annotated.mp4"),
        options=AnimationOptions(**base_options, reference_state=None, show_info_text=True),
        obstacles=obstacles,
    )


def print_summary(
    config: PaperRunConfig,
    params: CartPoleParams,
    goal_state: np.ndarray,
    plan: TrajectoryPlan,
    sim: SimulationResult,
    obstacles: list[Obstacle],
    switch_time: float | None,
) -> None:
    final_error = state_error(sim.state[-1], goal_state)
    print(f"[{config.name}] output: {config.output_dir}")
    print(f"[{config.name}] planned terminal state: {plan.state[-1]}")
    print(f"[{config.name}] displayed terminal state: {sim.state[-1]}")
    print(f"[{config.name}] final error norm: {np.linalg.norm(final_error):.6g}")
    print(f"[{config.name}] peak control: {np.max(np.abs(sim.control)):.6g} N")
    if switch_time is not None:
        print(f"[{config.name}] LQR switch time: {switch_time:.3f} s")
    if obstacles:
        clearance = minimum_obstacle_clearance(sim.state, params, obstacles, samples_per_link=config.obstacle_samples_per_link)
        print(f"[{config.name}] minimum obstacle clearance: {clearance:.6g} m")


def no_obstacle_case() -> PaperRunConfig:
    return PaperRunConfig(
        name="run1",
        output_dir=REPO_ROOT / "outputs/run1",
        track_bounds=(-1.5, 5.5),
        obstacles=(),
        stabilize=False,
    )


def paired_obstacle_case() -> PaperRunConfig:
    return PaperRunConfig(
        name="run2",
        output_dir=REPO_ROOT / "outputs/run2",
        track_bounds=(-2.5, 6.5),
        obstacles=tuple(default_track_obstacles()),
        stabilize=True,
    )


def goal_under_obstacle_case() -> PaperRunConfig:
    return PaperRunConfig(
        name="run3",
        output_dir=REPO_ROOT / "outputs/run3",
        track_bounds=(-2.5, 6.5),
        obstacles=tuple(goal_under_obstacle()),
        stabilize=True,
    )


def main_for(factory: Callable[[], PaperRunConfig]) -> None:
    run_case(factory())
