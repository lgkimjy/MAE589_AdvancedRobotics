from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from double_inverted_pendulum.controllers import RandomShootingMPCController
from double_inverted_pendulum.environment import DoubleInvertedPendulumEnv
from double_inverted_pendulum.model import default_params, downright_state, upright_state
from double_inverted_pendulum.visualization import AnimationOptions, animate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sampling-based MPC demo for the double inverted pendulum.")
    parser.add_argument("--save", type=str, default=None, help="Optional output path. Use .gif or .mp4.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--show-tip-trace", dest="show_tip_trace", action="store_true", help="Draw the end-tip trajectory.")
    parser.add_argument("--hide-tip-trace", dest="show_tip_trace", action="store_false", help="Hide the end-tip trajectory.")
    parser.add_argument("--show-history", dest="show_history", action="store_true", help="Draw fading history.")
    parser.add_argument("--hide-history", dest="show_history", action="store_false", help="Hide fading history.")
    parser.add_argument("--initial-angle-offset", type=float, default=0.08, help="Small offset from the exact downright pose in radians.")
    parser.add_argument("--goal-x", type=float, default=3.0, help="Target cart position.")
    parser.set_defaults(show_tip_trace=True, show_history=True)
    return parser.parse_args()


def main() -> None:
    temp_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    args = parse_args()

    params = default_params()
    track_bounds = (-2.5, 5.5)
    goal_state = upright_state()
    goal_state[0] = args.goal_x

    env = DoubleInvertedPendulumEnv(
        params=params,
        dt=0.04,
        t_final=8.0,
        cart_position_bounds=track_bounds,
        enforce_link_limits=True,
        initial_state=downright_state(args.initial_angle_offset),
    )
    controller = RandomShootingMPCController(
        params=params,
        equilibrium_state=goal_state,
        position_bounds=track_bounds,
    )
    sim = env.rollout(controller)

    print("Method:")
    print("Sampling MPC")
    print("\nGoal state:")
    print(goal_state)
    print("\nFinal state:")
    print(sim.state[-1])
    print("\nPeak control magnitude:")
    print(abs(sim.control).max())

    save_path = args.save
    if args.no_show and save_path is None:
        save_path = str(Path("outputs") / "mpc_demo.mp4")

    animate_simulation(
        sim_result=sim,
        params=params,
        show=not args.no_show,
        save_path=save_path,
        options=AnimationOptions(
            show_tip_trace=args.show_tip_trace,
            show_history=args.show_history,
            track_bounds=track_bounds,
            goal_x=goal_state[0],
        ),
        obstacles=[],
    )


if __name__ == "__main__":
    main()
