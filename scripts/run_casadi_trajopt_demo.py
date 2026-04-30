from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import site
import sys
import sysconfig
import tempfile

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
        # Keep the demo runnable from base shells that inherit a stale PYTHONPATH.
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

import numpy as np

from double_inverted_pendulum.collision import minimum_obstacle_clearance
from double_inverted_pendulum.controllers import (
    TrajectorySwitchingController,
    design_equilibrium_lqr,
    state_error,
)
from double_inverted_pendulum.environment import default_track_obstacles, goal_under_obstacle, staggered_track_obstacles
from double_inverted_pendulum.model import default_params, downright_state, upright_state
from double_inverted_pendulum.optimal_control import optimize_trajectory_with_casadi, optimize_trajectory_with_casadi_shooting
from double_inverted_pendulum.simulation import SimulationResult, rollout_open_loop
from double_inverted_pendulum.visualization import AnimationOptions, animate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CasADi-based nonlinear trajectory optimization demo.")
    parser.add_argument("--save", type=str, default=None, help="Optional output path. Use .gif or .mp4.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--show-tip-trace", dest="show_tip_trace", action="store_true", help="Draw the end-tip trajectory.")
    parser.add_argument("--hide-tip-trace", dest="show_tip_trace", action="store_false", help="Hide the end-tip trajectory.")
    parser.add_argument("--initial-angle-offset", type=float, default=0.0, help="Small offset from the exact downright pose in radians.")
    parser.add_argument("--horizon-steps", type=int, default=80, help="Number of multiple-shooting intervals.")
    parser.add_argument("--dt", type=float, default=0.04, help="Timestep for the optimal control discretization.")
    parser.add_argument("--hold-time", type=float, default=5.0, help="Extra simulation time after the planned trajectory to show terminal stabilization.")
    parser.add_argument("--goal-x", type=float, default=3.0, help="Target cart position.")
    parser.add_argument(
        "--optimizer",
        choices=("direct", "shooting"),
        default=None,
        help="Optimization backend. Defaults to direct multiple shooting.",
    )
    parser.add_argument(
        "--shooting-restarts",
        type=int,
        default=4,
        help="Number of deterministic initial guesses for the shooting optimizer.",
    )
    parser.add_argument(
        "--use-position-bounds",
        action="store_true",
        help="Enforce track bounds during the optimal control solve. Disabled by default because sqpmethod often struggles with the hard bounds.",
    )
    parser.add_argument(
        "--show-plan",
        dest="show_plan",
        action="store_true",
        help="Display the optimized state trajectory directly instead of replaying the control on the simulator.",
    )
    parser.add_argument(
        "--replay-plan",
        dest="show_plan",
        action="store_false",
        help="Replay the optimized open-loop control on the nonlinear simulator.",
    )
    parser.add_argument(
        "--hybrid-stabilize",
        dest="hybrid_stabilize",
        action="store_true",
        help="Replay the trajectory and use the selected feedback controller near the upright target.",
    )
    parser.add_argument(
        "--open-loop-replay",
        dest="hybrid_stabilize",
        action="store_false",
        help="Replay the optimized force profile without terminal feedback.",
    )
    parser.set_defaults(show_tip_trace=True, show_plan=None)
    parser.set_defaults(hybrid_stabilize=True)
    parser.add_argument(
        "--obstacle-avoidance",
        action="store_true",
        help="Add soft obstacle-avoidance penalties to the trajectory optimization problem.",
    )
    parser.add_argument(
        "--obstacle-layout",
        choices=("paired", "staggered", "goal-under"),
        default="paired",
        help="Obstacle placement used when obstacle avoidance is enabled.",
    )
    parser.add_argument(
        "--obstacle-clearance",
        type=float,
        default=0.08,
        help="Safety margin around each obstacle used by the soft collision penalty.",
    )
    parser.add_argument(
        "--obstacle-weight",
        type=float,
        default=450.0,
        help="Weight on the soft obstacle-avoidance penalty.",
    )
    parser.add_argument(
        "--obstacle-samples-per-link",
        type=int,
        default=4,
        help="Number of collision sample points per pendulum link.",
    )
    parser.add_argument(
        "--show-smoothstep-reference",
        dest="show_smoothstep_reference",
        action="store_true",
        help="Overlay the cubic smoothstep reference tip path (trajopt warm-start) on the animation.",
    )
    parser.add_argument(
        "--hide-smoothstep-reference",
        dest="show_smoothstep_reference",
        action="store_false",
        help="Do not draw the smoothstep reference curve.",
    )
    parser.set_defaults(show_smoothstep_reference=True)
    return parser.parse_args()


def build_obstacles(layout: str):
    if layout == "goal-under":
        return goal_under_obstacle()
    if layout == "staggered":
        return staggered_track_obstacles()
    return default_track_obstacles()


def print_casadi_diagnostics() -> None:
    try:
        import casadi as ca
    except ImportError:
        return

    print("\nCasADi diagnostics:")
    print(f"  version: {ca.__version__}")
    print(f"  path: {Path(ca.__file__).resolve()}")
    print(f"  IPOPT available: {ca.has_nlpsol('ipopt')}")
    print(f"  sqpmethod available: {ca.has_nlpsol('sqpmethod')}")


def main() -> None:
    temp_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    args = parse_args()

    params = replace(default_params(), 
        cart_damping=1.0,
        joint1_damping=0.1,
        joint2_damping=0.01,
        force_limit=150.0,
    )
    track_bounds = (-2.5, 6.5)
    initial_state = downright_state(args.initial_angle_offset)
    goal_state = upright_state()
    goal_state[0] = args.goal_x
    obstacles = build_obstacles(args.obstacle_layout) if args.obstacle_avoidance else []
    show_plan = bool(args.obstacle_avoidance if args.show_plan is None else args.show_plan)
    optimizer = args.optimizer or "direct"

    print(
        f"Solving trajectory optimization with {optimizer} backend "
        f"(N={args.horizon_steps}, dt={args.dt:.3f}, obstacles={len(obstacles)})...",
        flush=True,
    )
    try:
        optimizer_kwargs = dict(
            initial_state=initial_state,
            goal_state=goal_state,
            params=params,
            horizon_steps=args.horizon_steps,
            dt=args.dt,
            position_bounds=track_bounds if args.use_position_bounds else None,
            obstacles=obstacles,
            obstacle_clearance=args.obstacle_clearance,
            obstacle_weight=args.obstacle_weight,
            obstacle_samples_per_link=args.obstacle_samples_per_link,
        )
        if optimizer == "shooting":
            plan = optimize_trajectory_with_casadi_shooting(
                **optimizer_kwargs,
                restarts=args.shooting_restarts,
            )
        else:
            plan = optimize_trajectory_with_casadi(**optimizer_kwargs)
    except ImportError as exc:
        print(exc)
        print("Install the CasADi optional dependency first, then rerun this script.")
        return
    except RuntimeError as exc:
        print(exc)
        print_casadi_diagnostics()
        print(f"Interpreter: {Path(os.sys.executable)}")
        print("Tip: install and run with the same interpreter, for example:")
        print(f"  {os.sys.executable} -m pip install -e '.[casadi]'")
        print("For the direct backend, use a CasADi build with IPOPT. The PyPI wheel casadi>=3.7 usually includes it.")
        print("If an older CasADi is being imported from PYTHONPATH, try:")
        print(f"  env -u PYTHONPATH {os.sys.executable} scripts/run_casadi_trajopt_demo.py")
        print(f"  {os.sys.executable} scripts/run_casadi_trajopt_demo.py")
        return

    if show_plan:
        sim_result = SimulationResult(time=plan.time, state=plan.state, control=plan.control)
        if args.obstacle_avoidance:
            method_name = f"Obstacle-aware CasADi {optimizer} trajectory optimization (planned trajectory)"
        else:
            method_name = f"CasADi {optimizer} trajectory optimization (planned trajectory)"
        final_state = plan.state[-1]
    else:
        final_lqr = None
        if args.hybrid_stabilize:
            final_lqr = design_equilibrium_lqr(params=params, equilibrium_state=goal_state)
            replay_controller = TrajectorySwitchingController(
                params=params,
                plan_time=plan.time,
                plan_state=plan.state,
                plan_control=plan.control,
                equilibrium_state=goal_state,
                final_lqr=final_lqr,
            )
            method_name = "CasADi trajectory optimization + LQR stabilization"
        else:
            def open_loop_controller(time: float, _state) -> float:
                return float(np.interp(np.clip(time, plan.time[0], plan.time[-1]), plan.time, plan.control))

            replay_controller = open_loop_controller
            method_name = "CasADi nonlinear trajectory optimization (open-loop replay)"

        replay = rollout_open_loop(
            initial_state=initial_state,
            controller=replay_controller,
            params=params,
            t_final=float(plan.time[-1] + max(args.hold_time, 0.0)),
            dt=args.dt,
            position_bounds=track_bounds,
            enforce_link_limits=True,
        )
        sim_result = replay
        final_state = replay.state[-1]

    print("Method:")
    print(method_name)
    print("\nGoal state:")
    print(goal_state)
    print("\nPlanned terminal state:")
    print(plan.state[-1])
    print("\nPlanned trajectory duration [s]:")
    print(plan.time[-1])
    print("\nDisplayed terminal state:")
    print(final_state)
    if not show_plan and args.hybrid_stabilize:
        print("\nDisplayed simulation duration [s]:")
        print(sim_result.time[-1])
        wrapped_final_error = state_error(final_state, goal_state)
        print("\nController mode at end:")
        print(replay_controller.active_mode)
        print("\nFinal wrapped state error:")
        print(wrapped_final_error)
        print("\nFinal wrapped state error norm:")
        print(np.linalg.norm(wrapped_final_error))
        if final_lqr is not None:
            print("\nFinal LQR gain:")
            print(final_lqr.k)
    print("\nPeak control magnitude:")
    print(abs(sim_result.control).max())
    if obstacles:
        print("\nMinimum obstacle clearance [m]:")
        print(
            minimum_obstacle_clearance(
                sim_result.state,
                params,
                obstacles,
                samples_per_link=args.obstacle_samples_per_link,
            )
        )

    save_path = args.save
    if args.no_show and save_path is None:
        output_name = "trajopt_obstacle_avoidance_demo.mp4" if args.obstacle_avoidance else "casadi_trajopt_demo.mp4"
        save_path = str(Path("outputs") / output_name)

    if save_path is not None:
        print(f"\nRendering animation to {save_path}...", flush=True)
    elif not args.no_show:
        print("\nOpening animation window...", flush=True)

    animate_simulation(
        sim_result=sim_result,
        params=params,
        show=not args.no_show,
        save_path=save_path,
        options=AnimationOptions(
            show_tip_trace=args.show_tip_trace,
            track_bounds=track_bounds,
            goal_x=goal_state[0],
            reference_state=plan.reference_state if args.show_smoothstep_reference else None,
        ),
        obstacles=obstacles,
    )


if __name__ == "__main__":
    main()
