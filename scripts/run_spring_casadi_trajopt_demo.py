from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from spring_pendulum.model import default_params, downright_state, upright_state
from spring_pendulum.optimal_control import optimize_trajectory_with_casadi
from spring_pendulum.simulation import SimulationResult, rollout_open_loop
from spring_pendulum.visualization import AnimationOptions, animate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CasADi-based nonlinear trajectory optimization demo for the spring pendulum cart-pole.")
    parser.add_argument("--save", type=str, default=None, help="Optional output path. Use .gif or .mp4.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--show-tip-trace", dest="show_tip_trace", action="store_true", help="Draw the end-tip trajectory.")
    parser.add_argument("--hide-tip-trace", dest="show_tip_trace", action="store_false", help="Hide the end-tip trajectory.")
    parser.add_argument("--show-history", dest="show_history", action="store_true", help="Draw fading history.")
    parser.add_argument("--hide-history", dest="show_history", action="store_false", help="Hide fading history.")
    parser.add_argument("--initial-angle-offset", type=float, default=0.04, help="Small offset from the exact downright pose in radians.")
    parser.add_argument("--length-offset", type=float, default=0.0, help="Offset from hanging spring length in meters.")
    parser.add_argument("--horizon-steps", type=int, default=70, help="Number of multiple-shooting intervals.")
    parser.add_argument("--dt", type=float, default=0.06, help="Timestep for the optimal control discretization.")
    parser.add_argument("--goal-x", type=float, default=2.0, help="Target cart position.")
    parser.add_argument(
        "--use-position-bounds",
        action="store_true",
        help="Enforce track bounds during the optimal control solve.",
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
    parser.set_defaults(show_tip_trace=True, show_history=True, show_plan=True)
    return parser.parse_args()


def main() -> None:
    temp_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    args = parse_args()

    params = default_params()
    track_bounds = (-2.5, 4.5)
    initial_state = downright_state(args.initial_angle_offset, args.length_offset, params=params)
    goal_state = upright_state(params)
    goal_state[0] = args.goal_x

    try:
        plan = optimize_trajectory_with_casadi(
            initial_state=initial_state,
            goal_state=goal_state,
            params=params,
            horizon_steps=args.horizon_steps,
            dt=args.dt,
            position_bounds=track_bounds if args.use_position_bounds else None,
        )
    except ImportError as exc:
        print(exc)
        print("Install the CasADi optional dependency first, then rerun this script.")
        return
    except RuntimeError as exc:
        print(exc)
        print(f"Interpreter: {Path(os.sys.executable)}")
        print("Tip: install and run with the same interpreter, for example:")
        print(f"  {os.sys.executable} -m pip install -e '.[casadi]'")
        print(f"  {os.sys.executable} scripts/run_spring_casadi_trajopt_demo.py")
        return

    if args.show_plan:
        sim_result = SimulationResult(time=plan.time, state=plan.state, control=plan.control)
        method_name = "Spring pendulum CasADi trajectory optimization (planned trajectory)"
        final_state = plan.state[-1]
    else:
        def open_loop_controller(time: float, _state) -> float:
            idx = min(int(round(time / args.dt)), plan.control.size - 1)
            return float(plan.control[idx])

        replay = rollout_open_loop(
            initial_state=initial_state,
            controller=open_loop_controller,
            params=params,
            t_final=plan.time[-1],
            dt=args.dt,
            position_bounds=track_bounds,
            enforce_link_limits=True,
        )
        sim_result = replay
        method_name = "Spring pendulum CasADi trajectory optimization (open-loop replay)"
        final_state = replay.state[-1]

    print("Method:")
    print(method_name)
    print("\nGoal state:")
    print(goal_state)
    print("\nPlanned terminal state:")
    print(plan.state[-1])
    print("\nDisplayed terminal state:")
    print(final_state)
    print("\nPeak control magnitude:")
    print(abs(sim_result.control).max())

    save_path = args.save
    if args.no_show and save_path is None:
        save_path = str(Path("outputs") / "spring_casadi_trajopt_demo.mp4")

    animate_simulation(
        sim_result=sim_result,
        params=params,
        show=not args.no_show,
        save_path=save_path,
        options=AnimationOptions(
            show_tip_trace=args.show_tip_trace,
            show_history=args.show_history,
            track_bounds=track_bounds,
            goal_x=goal_state[0],
        ),
    )


if __name__ == "__main__":
    main()
