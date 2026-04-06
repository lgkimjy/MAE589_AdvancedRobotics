from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import tempfile

from double_inverted_pendulum.environment import BoxObstacle, DoubleInvertedPendulumEnv
from double_inverted_pendulum.model import default_params, downright_state, upright_state
from double_inverted_pendulum.optimal_control import MPPIController
from double_inverted_pendulum.visualization import AnimationOptions, animate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPPI demo for the double inverted pendulum.")
    parser.add_argument("--save", type=str, default=None, help="Optional output path. Use .gif or .mp4.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--show-tip-trace", dest="show_tip_trace", action="store_true", help="Draw the end-tip trajectory.")
    parser.add_argument("--hide-tip-trace", dest="show_tip_trace", action="store_false", help="Hide the end-tip trajectory.")
    parser.add_argument("--show-prediction", dest="show_prediction", action="store_true", help="Draw the MPPI predicted rollout at each frame.")
    parser.add_argument("--hide-prediction", dest="show_prediction", action="store_false", help="Hide the MPPI predicted rollout.")
    parser.add_argument("--initial-angle-offset", type=float, default=0.00, help="Small offset from the exact downright pose in radians.")
    parser.add_argument("--sim-time", type=float, default=40.0, help="Simulation time for swing-up and terminal settling.")
    parser.set_defaults(show_tip_trace=True, show_prediction=True)
    return parser.parse_args()


def main() -> None:
    temp_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    args = parse_args()

    params = replace(
        default_params(), 
        cart_damping=0.0,
        joint1_damping=0.03,
        joint2_damping=0.03,
        force_limit=50.0,
    )
    track_bounds = (-1.5, 7.5)
    goal_state = upright_state()
    goal_state[0] = 3.0

    env = DoubleInvertedPendulumEnv(
        params=params,
        dt=0.05,
        t_final=args.sim_time,
        cart_position_bounds=track_bounds,
        enforce_link_limits=True,
        initial_state=downright_state(args.initial_angle_offset),
    )
    obstacles = [
        BoxObstacle(center=(1.4, 0.55), width=0.75, height=0.4, color="#d66853"),
        BoxObstacle(center=(1.4, -0.55), width=0.75, height=0.4, color="#d66853"),
    ]
    env.obstacles = obstacles

    controller = MPPIController(
        params=params,
        goal_state=goal_state,
        dt=env.dt,
        horizon=24,
        num_samples=5000,
        noise_sigma=2.0,
        temperature=10.0,
        action_repeat=1,
        position_bounds=track_bounds,
    )
    sim = env.rollout(controller)

    print("Method:")
    print("MPPI")
    print("\nGoal state:")
    print(goal_state)
    print("\nFinal state:")
    print(sim.state[-1])
    print("\nPeak control magnitude:")
    print(abs(sim.control).max())

    save_path = args.save
    if args.no_show and save_path is None:
        save_path = str(Path("outputs") / "mppi_demo.mp4")

    animate_simulation(
        sim_result=sim,
        params=params,
        show=not args.no_show,
        save_path=save_path,
        options=AnimationOptions(
            show_tip_trace=args.show_tip_trace,
            show_prediction=args.show_prediction,
            track_bounds=track_bounds,
            goal_x=goal_state[0],
        ),
        obstacles=obstacles,
    )


if __name__ == "__main__":
    main()
