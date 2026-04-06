from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import tempfile

import numpy as np

from double_inverted_pendulum.environment import DoubleInvertedPendulumEnv
from double_inverted_pendulum.model import default_params
from double_inverted_pendulum.visualization import AnimationOptions, animate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive double-pendulum demo with zero control input.")
    parser.add_argument("--save", type=str, default=None, help="Optional output path. Use .gif or .mp4.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--show-tip-trace", dest="show_tip_trace", action="store_true", help="Draw the end-tip trajectory.")
    parser.add_argument("--hide-tip-trace", dest="show_tip_trace", action="store_false", help="Hide the end-tip trajectory.")
    parser.add_argument("--sim-time", type=float, default=60.0, help="Simulation time for the passive motion.")
    parser.add_argument("--theta1", type=float, default=0.0, help="Initial angle of the first link [rad].")
    parser.add_argument("--theta2", type=float, default=0.0, help="Initial angle of the second link [rad].")
    parser.add_argument("--theta1-dot", dest="theta1_dot", type=float, default=1.8, help="Initial angular velocity of the first link [rad/s].")
    parser.add_argument("--theta2-dot", dest="theta2_dot", type=float, default=-1.15, help="Initial angular velocity of the second link [rad/s].")
    parser.set_defaults(show_tip_trace=True)
    return parser.parse_args()


def main() -> None:
    temp_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    args = parse_args()

    params = replace(
        default_params(),
        cart_damping=5000.0,
        joint1_damping=0.0,
        joint2_damping=0.0,
    )
    track_bounds = (-3.0, 3.0)
    initial_state = np.array([0.0, args.theta1, args.theta2, 0.0, args.theta1_dot, args.theta2_dot], dtype=float)
    env = DoubleInvertedPendulumEnv(
        params=params,
        dt=0.02,
        t_final=args.sim_time,
        cart_position_bounds=track_bounds,
        enforce_link_limits=False,
        initial_state=initial_state,
    )
    sim = env.rollout()

    print("Method:")
    print("Passive dynamics (zero input)")
    print("\nInitial state:")
    print(initial_state)

    save_path = args.save
    if args.no_show and save_path is None:
        save_path = str(Path("outputs") / "chaos_demo.mp4")

    animate_simulation(
        sim_result=sim,
        params=params,
        show=not args.no_show,
        save_path=save_path,
        options=AnimationOptions(
            show_tip_trace=args.show_tip_trace,
            trace_window=1200000,
            track_bounds=track_bounds,
        ),
        obstacles=[],
    )


if __name__ == "__main__":
    main()
