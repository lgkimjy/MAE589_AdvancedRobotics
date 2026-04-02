"""Tools for the double inverted pendulum cart-pole problem."""

from .controllers import (
    GainScheduledLQRController,
    LQRGain,
    RandomShootingMPCController,
    lqr_gain,
    wrap_to_pi,
)
from .dynamics import cartpole_dynamics, linear_state_space
from .environment import CircularObstacle, DoubleInvertedPendulumEnv, default_track_obstacles
from .kinematics import cart_and_tip_positions, end_effector_path, horizontal_span
from .model import CartPoleParams, default_params, downright_state, upright_state
from .optimal_control import (
    MPPIController,
    TrajectoryPlan,
    optimize_trajectory_with_casadi,
)
from .simulation import SimulationResult, rollout_open_loop, simulate_closed_loop
from .visualization import AnimationOptions, animate_simulation

__all__ = [
    "AnimationOptions",
    "CartPoleParams",
    "CircularObstacle",
    "DoubleInvertedPendulumEnv",
    "default_track_obstacles",
    "GainScheduledLQRController",
    "LQRGain",
    "MPPIController",
    "RandomShootingMPCController",
    "SimulationResult",
    "TrajectoryPlan",
    "animate_simulation",
    "cart_and_tip_positions",
    "cartpole_dynamics",
    "default_params",
    "downright_state",
    "end_effector_path",
    "horizontal_span",
    "linear_state_space",
    "lqr_gain",
    "optimize_trajectory_with_casadi",
    "rollout_open_loop",
    "simulate_closed_loop",
    "upright_state",
    "wrap_to_pi",
]
