from __future__ import annotations

import argparse
import os
import numpy as np
from pathlib import Path
import tempfile

from double_inverted_pendulum.controllers import GainScheduledLQRController
from double_inverted_pendulum.environment import DoubleInvertedPendulumEnv
from double_inverted_pendulum.model import default_params, downright_state, upright_state
from double_inverted_pendulum.visualization import AnimationOptions, animate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gain-scheduled LQR demo for the double inverted pendulum.")
    parser.add_argument("--save", type=str, default=None, help="Optional output path. Use .gif or .mp4.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--show-tip-trace", dest="show_tip_trace", action="store_true", help="Draw the end-tip trajectory.")
    parser.add_argument("--hide-tip-trace", dest="show_tip_trace", action="store_false", help="Hide the end-tip trajectory.")
    parser.add_argument("--initial-angle-offset", type=float, default=0.08, help="Small offset from the exact downright pose in radians.")
    parser.set_defaults(show_tip_trace=True)
    return parser.parse_args()


def main() -> None:
    temp_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    args = parse_args()
    params = default_params()
    track_bounds = (-2.5, 5.5)
    x_goal = upright_state()
    x_goal[0] = 3.0
    controller = GainScheduledLQRController(
        params=params,
        target_position=x_goal[0],
    )

    initial_state = downright_state(args.initial_angle_offset)
    env = DoubleInvertedPendulumEnv(
        params=params,
        dt=0.02,
        t_final=7.0,
        cart_position_bounds=track_bounds,
        enforce_link_limits=True,
        initial_state=initial_state,
    )
    sim = env.rollout(controller)

    print("Goal state (upright):")
    print(x_goal)
    print("\nInitial state:")
    print(initial_state)
    print("\nTrack bounds:")
    print(track_bounds)
    print("\nScheduled nodes:")
    print(len(controller.nodes))
    print("\nFirst node reference / input:")
    print(controller.nodes[0].reference_state, controller.nodes[0].reference_input)
    print("\nLast node reference / input:")
    print(controller.nodes[-1].reference_state, controller.nodes[-1].reference_input)
    print("\nFinal state:")
    print(sim.state[-1])
    print("\nPeak control magnitude:")
    print(np.max(np.abs(sim.control)))
    print("\nFinal state error norm:")
    print(np.linalg.norm(sim.state[-1] - x_goal))

    save_path = args.save
    if args.no_show and save_path is None:
        save_path = str(Path("outputs") / "lqr_upright_demo.mp4")

    animate_simulation(
        sim_result=sim,
        params=params,
        show=not args.no_show,
        save_path=save_path,
        options=AnimationOptions(
            show_tip_trace=args.show_tip_trace,
            track_bounds=track_bounds,
            goal_x=x_goal[0],
        ),
        obstacles=[],
    )


if __name__ == "__main__":
    main()
