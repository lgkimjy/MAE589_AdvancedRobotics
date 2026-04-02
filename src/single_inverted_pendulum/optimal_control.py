from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from double_inverted_pendulum.controllers import wrap_to_pi

from .dynamics import cartpole_dynamics
from .model import SingleCartPoleParams


@dataclass
class MPPIController:
    params: SingleCartPoleParams
    goal_state: np.ndarray
    dt: float = 0.05
    horizon: int = 28
    num_samples: int = 256
    noise_sigma: float = 7.0
    temperature: float = 4.0
    action_clip: float | None = None
    position_bounds: tuple[float, float] | None = None
    action_repeat: int = 1
    nominal_sequence: np.ndarray = field(default_factory=lambda: np.zeros(28, dtype=float))
    last_predicted_states: np.ndarray | None = field(default=None, init=False)
    last_sampled_states: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.goal_state = np.asarray(self.goal_state, dtype=float)
        if self.action_clip is None:
            self.action_clip = self.params.force_limit
        if self.nominal_sequence.shape[0] != self.horizon:
            self.nominal_sequence = np.zeros(self.horizon, dtype=float)
        self._rng = np.random.default_rng(13)

    def __call__(self, _time: float, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        coarse_horizon = int(np.ceil(self.horizon / max(1, self.action_repeat)))
        coarse_nominal = self.nominal_sequence[:: max(1, self.action_repeat)][:coarse_horizon]
        noise = self._rng.normal(0.0, self.noise_sigma, size=(self.num_samples, coarse_horizon))
        candidate_coarse = np.clip(coarse_nominal[None, :] + noise, -self.action_clip, self.action_clip)
        candidate_u = np.repeat(candidate_coarse, max(1, self.action_repeat), axis=1)[:, : self.horizon]
        sampled_states = self._predict_states_batch(state, candidate_u)
        costs = self._trajectory_cost_batch(sampled_states, candidate_u)

        beta = float(np.min(costs))
        weights = np.exp(-(costs - beta) / max(self.temperature, 1e-6))
        weights /= np.sum(weights)
        self.nominal_sequence = np.sum(weights[:, None] * candidate_u, axis=0)
        self.last_predicted_states = self._predict_states(state, self.nominal_sequence)
        sample_stride = max(1, self.num_samples // 64)
        self.last_sampled_states = sampled_states[::sample_stride]
        action = float(np.clip(self.nominal_sequence[0], -self.action_clip, self.action_clip))
        self.nominal_sequence[:-1] = self.nominal_sequence[1:]
        self.nominal_sequence[-1] = 0.0
        return action

    def _trajectory_cost_batch(self, rollout: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = rollout[:, 1:, :]
        angle_error = wrap_to_pi(states[:, :, 1] - self.goal_state[1])
        x_error = states[:, :, 0] - self.goal_state[0]
        x_dot = states[:, :, 2]
        theta_dot = states[:, :, 3]
        progress = np.clip((np.cos(states[:, :, 1]) + 1.0) / 2.0, 0.0, 1.0)

        cost = np.sum((0.15 + 2.8 * progress) * x_error**2, axis=1)
        cost += np.sum(32.0 * angle_error**2, axis=1)
        cost += np.sum(0.10 * x_dot**2 + 0.20 * theta_dot**2, axis=1)
        cost += np.sum(0.0012 * controls**2, axis=1)
        cost -= np.sum(12.0 * np.cos(states[:, :, 1]), axis=1)
        cost += np.sum(5.0 * progress * x_error**2, axis=1)

        if self.position_bounds is not None:
            lower_margin = states[:, :, 0] - self.position_bounds[0]
            upper_margin = self.position_bounds[1] - states[:, :, 0]
            cost += np.sum(140.0 * np.maximum(0.0, 0.25 - lower_margin) ** 2, axis=1)
            cost += np.sum(140.0 * np.maximum(0.0, 0.25 - upper_margin) ** 2, axis=1)

        tail = rollout[:, -min(8, rollout.shape[1] - 1) :, :]
        tail_angle = wrap_to_pi(tail[:, :, 1] - self.goal_state[1])
        cost += np.sum(50.0 * (tail[:, :, 0] - self.goal_state[0]) ** 2, axis=1)
        cost += np.sum(140.0 * tail_angle**2, axis=1)
        cost += np.sum(3.5 * tail[:, :, 2] ** 2 + 5.0 * tail[:, :, 3] ** 2, axis=1)

        terminal = rollout[:, -1, :]
        terminal_angle = wrap_to_pi(terminal[:, 1] - self.goal_state[1])
        cost += 220.0 * (terminal[:, 0] - self.goal_state[0]) ** 2
        cost += 600.0 * terminal_angle**2
        cost += 8.0 * terminal[:, 2] ** 2 + 14.0 * terminal[:, 3] ** 2
        return cost.astype(float)

    def _predict_states(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float).copy()
        rollout = np.zeros((controls.shape[0] + 1, x.size), dtype=float)
        rollout[0] = x
        for idx, u in enumerate(controls, start=1):
            x = self._rk4_step(x, float(u))
            if self.position_bounds is not None:
                x[0] = np.clip(x[0], self.position_bounds[0], self.position_bounds[1])
            rollout[idx] = x
        return rollout

    def _predict_states_batch(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        x = np.repeat(np.asarray(state, dtype=float)[None, :], controls.shape[0], axis=0)
        rollout = np.zeros((controls.shape[0], controls.shape[1] + 1, x.shape[1]), dtype=float)
        rollout[:, 0, :] = x
        for idx in range(controls.shape[1]):
            x = self._rk4_step_batch(x, controls[:, idx])
            if self.position_bounds is not None:
                x[:, 0] = np.clip(x[:, 0], self.position_bounds[0], self.position_bounds[1])
            rollout[:, idx + 1, :] = x
        return rollout

    def _rk4_step(self, state: np.ndarray, control: float) -> np.ndarray:
        k1 = cartpole_dynamics(0.0, state, control, self.params)
        k2 = cartpole_dynamics(0.0, state + 0.5 * self.dt * k1, control, self.params)
        k3 = cartpole_dynamics(0.0, state + 0.5 * self.dt * k2, control, self.params)
        k4 = cartpole_dynamics(0.0, state + self.dt * k3, control, self.params)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_step_batch(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        k1 = self._dynamics_batch(state, control)
        k2 = self._dynamics_batch(state + 0.5 * self.dt * k1, control)
        k3 = self._dynamics_batch(state + 0.5 * self.dt * k2, control)
        k4 = self._dynamics_batch(state + self.dt * k3, control)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _dynamics_batch(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        theta = state[:, 1]
        x_dot = state[:, 2]
        theta_dot = state[:, 3]
        clipped = np.clip(control, -self.params.force_limit, self.params.force_limit)

        m0 = self.params.cart_mass
        m1 = self.params.pole_mass
        l = self.params.pole_length
        g = self.params.gravity

        mass_matrix = np.empty((state.shape[0], 2, 2), dtype=float)
        mass_matrix[:, 0, 0] = m0 + m1
        mass_matrix[:, 0, 1] = m1 * l * np.cos(theta)
        mass_matrix[:, 1, 0] = mass_matrix[:, 0, 1]
        mass_matrix[:, 1, 1] = m1 * l**2

        generalized = np.empty((state.shape[0], 2), dtype=float)
        generalized[:, 0] = clipped - self.params.cart_damping * x_dot + m1 * l * np.sin(theta) * theta_dot**2
        generalized[:, 1] = -self.params.joint_damping * theta_dot + m1 * g * l * np.sin(theta)
        accelerations = np.linalg.solve(mass_matrix, generalized[..., None])[..., 0]
        return np.column_stack((x_dot, theta_dot, accelerations))
