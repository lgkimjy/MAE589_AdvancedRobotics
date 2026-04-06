from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import tempfile

import numpy as np

from double_inverted_pendulum.controllers import TrajectorySwitchingController, design_equilibrium_lqr, state_error
from double_inverted_pendulum.model import default_params, downright_state, upright_state
from double_inverted_pendulum.optimal_control import optimize_trajectory_with_casadi
from double_inverted_pendulum.simulation import SimulationResult, rollout_open_loop
from double_inverted_pendulum.visualization import AnimationOptions, animate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CasADi-based nonlinear trajectory optimization demo.")
    parser.add_argument("--save", type=str, default=None, help="Optional output path. Use .gif or .mp4.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--show-tip-trace", dest="show_tip_trace", action="store_true", help="Draw the end-tip trajectory.")
    parser.add_argument("--hide-tip-trace", dest="show_tip_trace", action="store_false", help="Hide the end-tip trajectory.")
    parser.add_argument("--initial-angle-offset", type=float, default=0.0, help="Small offset from the exact downright pose in radians.")
    parser.add_argument("--horizon-steps", type=int, default=360, help="Number of multiple-shooting intervals.")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep for the optimal control discretization.")
    parser.add_argument("--hold-time", type=float, default=5.0, help="Extra simulation time after the planned trajectory to show final LQR stabilization.")
    parser.add_argument("--goal-x", type=float, default=3.0, help="Target cart position.")
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
        help="Replay the trajectory and switch to a final equilibrium LQR near the upright target.",
    )
    parser.add_argument(
        "--open-loop-replay",
        dest="hybrid_stabilize",
        action="store_false",
        help="Replay the optimized force profile without the final equilibrium LQR.",
    )
    parser.set_defaults(show_tip_trace=True, show_plan=False)
    parser.set_defaults(hybrid_stabilize=True)
    return parser.parse_args()


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
    track_bounds = (-2.5, 7.5)
    initial_state = downright_state(args.initial_angle_offset)
    goal_state = upright_state()
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
        print(f"  {os.sys.executable} scripts/run_casadi_trajopt_demo.py")
        return

    if args.show_plan:
        sim_result = SimulationResult(time=plan.time, state=plan.state, control=plan.control)
        method_name = "CasADi nonlinear trajectory optimization (planned trajectory)"
        final_state = plan.state[-1]
    else:
        final_lqr = design_equilibrium_lqr(params=params, equilibrium_state=goal_state)
        if args.hybrid_stabilize:
            replay_controller = TrajectorySwitchingController(
                params=params,
                plan_time=plan.time,
                plan_state=plan.state,
                plan_control=plan.control,
                equilibrium_state=goal_state,
                final_lqr=final_lqr,
            )
            method_name = "CasADi trajectory optimization + final equilibrium LQR"
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
    if not args.show_plan and args.hybrid_stabilize:
        print("\nDisplayed simulation duration [s]:")
        print(sim_result.time[-1])
        wrapped_final_error = state_error(final_state, goal_state)
        print("\nController mode at end:")
        print(replay_controller.active_mode)
        print("\nFinal wrapped state error:")
        print(wrapped_final_error)
        print("\nFinal wrapped state error norm:")
        print(np.linalg.norm(wrapped_final_error))
        print("\nFinal LQR gain:")
        print(final_lqr.k)
    print("\nPeak control magnitude:")
    print(abs(sim_result.control).max())

    save_path = args.save
    if args.no_show and save_path is None:
        save_path = str(Path("outputs") / "casadi_trajopt_demo.mp4")

    animate_simulation(
        sim_result=sim_result,
        params=params,
        show=not args.no_show,
        save_path=save_path,
        options=AnimationOptions(
            show_tip_trace=args.show_tip_trace,
            track_bounds=track_bounds,
            goal_x=goal_state[0],
        ),
        obstacles=[],
    )


if __name__ == "__main__":
    main()
