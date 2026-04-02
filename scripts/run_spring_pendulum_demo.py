from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from spring_pendulum.environment import SpringPendulumEnv
from spring_pendulum.model import default_params, downright_state
from spring_pendulum.visualization import AnimationOptions, animate_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive demo for the spring pendulum cart-pole.")
    parser.add_argument("--save", type=str, default=None, help="Optional output path. Use .gif or .mp4.")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display.")
    parser.add_argument("--initial-angle-offset", type=float, default=0.1, help="Offset from the hanging pose in radians.")
    parser.add_argument("--length-offset", type=float, default=0.0, help="Offset from the hanging spring length in meters.")
    parser.add_argument("--cart-offset", type=float, default=0.0, help="Initial cart position offset in meters.")
    parser.add_argument("--sim-time", type=float, default=12.0, help="Simulation time in seconds.")
    parser.set_defaults()
    return parser.parse_args()


def main() -> None:
    temp_cache_dir = Path(tempfile.gettempdir()) / "matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(temp_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
    args = parse_args()

    params = default_params()
    initial_state = downright_state(
        angle_perturbation=args.initial_angle_offset,
        length_offset=args.length_offset,
        params=params,
    )
    initial_state[0] = args.cart_offset
    env = SpringPendulumEnv(
        params=params,
        dt=0.02,
        t_final=args.sim_time,
        cart_position_bounds=(-2.5, 4.5),
        initial_state=initial_state,
    )
    sim = env.rollout()

    print("Method:")
    print("Passive Spring Pendulum Cart-Pole")
    print("\nInitial state:")
    print(env.initial_state)
    print("\nFinal state:")
    print(sim.state[-1])
    print("\nObserved spring length range:")
    print((sim.state[:, 2].min(), sim.state[:, 2].max()))

    save_path = args.save
    if args.no_show and save_path is None:
        save_path = str(Path("outputs") / "spring_pendulum_demo.mp4")

    animate_simulation(
        sim_result=sim,
        params=params,
        show=not args.no_show,
        save_path=save_path,
        options=AnimationOptions(
            show_tip_trace=True,
            show_history=True,
            track_bounds=env.cart_position_bounds,
        ),
    )


if __name__ == "__main__":
    main()
