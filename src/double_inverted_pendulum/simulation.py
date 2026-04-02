from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .controllers import LQRGain
from .dynamics import cartpole_dynamics
from .kinematics import horizontal_span
from .model import CartPoleParams


@dataclass(frozen=True)
class SimulationResult:
    time: np.ndarray
    state: np.ndarray
    control: np.ndarray
    predicted_state: np.ndarray | None = None
    predicted_samples: np.ndarray | None = None


ControlLaw = Callable[[float, np.ndarray], float]


def rollout_open_loop(
    initial_state: np.ndarray,
    controller: ControlLaw,
    params: CartPoleParams,
    t_final: float = 10.0,
    dt: float = 0.01,
    position_bounds: tuple[float, float] | None = None,
    enforce_link_limits: bool = True,
) -> SimulationResult:
    x0 = np.asarray(initial_state, dtype=float)
    time_grid = np.arange(0.0, t_final + dt, dt)
    state_hist = np.zeros((time_grid.size, x0.size), dtype=float)
    control_hist = np.zeros(time_grid.size, dtype=float)
    state_hist[0] = x0
    predicted_hist: list[np.ndarray] = []
    predicted_sample_hist: list[np.ndarray] = []

    def clipped_control(t: float, state: np.ndarray) -> float:
        raw = float(controller(t, state))
        return float(np.clip(raw, -params.force_limit, params.force_limit))

    for idx in range(time_grid.size - 1):
        t_now = time_grid[idx]
        state_now = state_hist[idx]
        control_now = clipped_control(t_now, state_now)
        control_hist[idx] = control_now
        prediction = getattr(controller, "last_predicted_states", None)
        prediction_samples = getattr(controller, "last_sampled_states", None)
        predicted_hist.append(None if prediction is None else np.asarray(prediction, dtype=float).copy())
        predicted_sample_hist.append(
            None if prediction_samples is None else np.asarray(prediction_samples, dtype=float).copy()
        )

        sol = solve_ivp(
            lambda t, state: cartpole_dynamics(t, state, control_now, params),
            (t_now, time_grid[idx + 1]),
            state_now,
            t_eval=[time_grid[idx + 1]],
            rtol=1e-7,
            atol=1e-9,
        )
        state_hist[idx + 1] = sol.y[:, -1]
        if position_bounds is not None:
            x_min, x_max = position_bounds
            min_x, max_x = horizontal_span(state_hist[idx + 1], params)
            if state_hist[idx + 1, 0] > x_max:
                state_hist[idx + 1, 0] = x_max
                state_hist[idx + 1, 3] = 0.0
            elif state_hist[idx + 1, 0] < x_min:
                state_hist[idx + 1, 0] = x_min
                state_hist[idx + 1, 3] = 0.0
            elif enforce_link_limits and (max_x > x_max or min_x < x_min):
                state_hist[idx + 1] = state_hist[idx].copy()
                state_hist[idx + 1, 3:] = 0.0

    control_hist[-1] = clipped_control(time_grid[-1], state_hist[-1])
    final_prediction = getattr(controller, "last_predicted_states", None)
    final_prediction_samples = getattr(controller, "last_sampled_states", None)
    predicted_hist.append(None if final_prediction is None else np.asarray(final_prediction, dtype=float).copy())
    predicted_sample_hist.append(
        None if final_prediction_samples is None else np.asarray(final_prediction_samples, dtype=float).copy()
    )

    available_predictions = [pred for pred in predicted_hist if pred is not None]
    if available_predictions:
        horizon = max(pred.shape[0] for pred in available_predictions)
        pred_dim = available_predictions[0].shape[1]
        predicted_state = np.full((time_grid.size, horizon, pred_dim), np.nan, dtype=float)
        for idx, pred in enumerate(predicted_hist):
            if pred is None:
                continue
            predicted_state[idx, : pred.shape[0], :] = pred
    else:
        predicted_state = None

    available_sample_predictions = [pred for pred in predicted_sample_hist if pred is not None]
    if available_sample_predictions:
        max_samples = max(pred.shape[0] for pred in available_sample_predictions)
        max_horizon = max(pred.shape[1] for pred in available_sample_predictions)
        pred_dim = available_sample_predictions[0].shape[2]
        predicted_samples = np.full((time_grid.size, max_samples, max_horizon, pred_dim), np.nan, dtype=float)
        for idx, pred in enumerate(predicted_sample_hist):
            if pred is None:
                continue
            predicted_samples[idx, : pred.shape[0], : pred.shape[1], :] = pred
    else:
        predicted_samples = None

    return SimulationResult(
        time=time_grid,
        state=state_hist,
        control=control_hist,
        predicted_state=predicted_state,
        predicted_samples=predicted_samples,
    )


def simulate_closed_loop(
    initial_state: np.ndarray,
    controller: LQRGain,
    params: CartPoleParams,
    t_final: float = 10.0,
    dt: float = 0.01,
    equilibrium_state: np.ndarray | None = None,
    position_bounds: tuple[float, float] | None = None,
    enforce_link_limits: bool = True,
) -> SimulationResult:
    x_eq = np.zeros(6, dtype=float) if equilibrium_state is None else np.asarray(equilibrium_state, dtype=float)
    def control_law(state: np.ndarray) -> float:
        delta = state - x_eq
        raw = float(-(controller.k @ delta.reshape(-1, 1)).item())
        return float(np.clip(raw, -params.force_limit, params.force_limit))

    return rollout_open_loop(
        initial_state=initial_state,
        controller=lambda _t, state: control_law(state),
        params=params,
        t_final=t_final,
        dt=dt,
        position_bounds=position_bounds,
        enforce_link_limits=enforce_link_limits,
    )
