from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize_scalar

from .dynamics import cartpole_dynamics, linear_state_space
from .model import CartPoleParams


@dataclass(frozen=True)
class LQRGain:
    k: np.ndarray
    s: np.ndarray
    closed_loop_eigs: np.ndarray


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class RandomShootingMPCController:
    params: CartPoleParams
    equilibrium_state: np.ndarray
    lqr: LQRGain | None = None
    action_set: tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
    horizon_blocks: int = 3
    block_steps: int = 5
    predict_dt: float = 0.04
    switch_angle: float = 0.28
    switch_rate: float = 1.2
    position_bounds: tuple[float, float] | None = None
    _last_index: int = field(default=-1, init=False)
    _current_action: float = field(default=0.0, init=False)
    _sequences: list[tuple[float, ...]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.equilibrium_state = np.asarray(self.equilibrium_state, dtype=float)
        self._sequences = list(product(self.action_set, repeat=self.horizon_blocks))

    def __call__(self, time: float, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        angle_error = wrap_to_pi(state[1:3] - self.equilibrium_state[1:3])
        local_delta = np.array(
            [
                state[0] - self.equilibrium_state[0],
                angle_error[0],
                angle_error[1],
                state[3] - self.equilibrium_state[3],
                state[4] - self.equilibrium_state[4],
                state[5] - self.equilibrium_state[5],
            ],
            dtype=float,
        )

        if self.lqr is not None and np.max(np.abs(angle_error)) < self.switch_angle and np.linalg.norm(local_delta[3:]) < self.switch_rate:
            return float(-(self.lqr.k @ local_delta.reshape(-1, 1)).item())

        step_index = int(np.floor(max(time, 0.0) / self.predict_dt + 1e-9))
        if step_index != self._last_index:
            self._last_index = step_index
            self._current_action = self._plan_action(state)
        return self._current_action

    def _plan_action(self, state: np.ndarray) -> float:
        best_cost = None
        best_action = 0.0
        for seq in self._sequences:
            rollout = self._predict_rollout(state, seq)
            cost = self._trajectory_cost(rollout, seq)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_action = seq[0]
        return float(best_action)

    def _predict_rollout(self, state: np.ndarray, action_sequence: tuple[float, ...]) -> np.ndarray:
        rollout = np.zeros((self.horizon_blocks * self.block_steps + 1, state.size), dtype=float)
        rollout[0] = state
        current = state.copy()
        cursor = 1
        for action in action_sequence:
            for _ in range(self.block_steps):
                current = current + self.predict_dt * cartpole_dynamics(0.0, current, action, self.params)
                if self.position_bounds is not None:
                    current[0] = np.clip(current[0], self.position_bounds[0], self.position_bounds[1])
                rollout[cursor] = current
                cursor += 1
        return rollout

    def _trajectory_cost(self, rollout: np.ndarray, action_sequence: tuple[float, ...]) -> float:
        goal_x = self.equilibrium_state[0]
        cost = 0.0
        for state in rollout[:-1]:
            angle1 = float(wrap_to_pi(state[1] - self.equilibrium_state[1]))
            angle2 = float(wrap_to_pi(state[2] - self.equilibrium_state[2]))
            tip_height_reward = np.cos(state[1]) + 0.7 * np.cos(state[2])
            cost += 0.6 * (state[0] - goal_x) ** 2
            cost += 5.0 * angle1**2 + 4.0 * angle2**2
            cost += 0.15 * state[3] ** 2 + 0.08 * state[4] ** 2 + 0.08 * state[5] ** 2
            cost -= 2.5 * tip_height_reward
            if self.position_bounds is not None:
                if state[0] <= self.position_bounds[0] + 0.05 or state[0] >= self.position_bounds[1] - 0.05:
                    cost += 15.0
        terminal = rollout[-1]
        angle1 = float(wrap_to_pi(terminal[1] - self.equilibrium_state[1]))
        angle2 = float(wrap_to_pi(terminal[2] - self.equilibrium_state[2]))
        cost += 6.0 * (terminal[0] - goal_x) ** 2
        cost += 40.0 * angle1**2 + 35.0 * angle2**2
        cost += 0.01 * sum(u * u for u in action_sequence)
        return float(cost)


@dataclass(frozen=True)
class ScheduledLQRNode:
    reference_state: np.ndarray
    reference_input: float
    gain: LQRGain


@dataclass
class GainScheduledLQRController:
    params: CartPoleParams
    target_position: float = 3.0
    n_nodes: int = 9
    q_mat: np.ndarray = field(
        default_factory=lambda: np.diag([25.0, 180.0, 180.0, 10.0, 22.0, 22.0]).astype(float)
    )
    r_mat: np.ndarray = field(default_factory=lambda: np.array([[0.9]], dtype=float))
    angle_weight: float = 3.0
    velocity_weight: float = 0.35
    position_weight: float = 0.9
    nodes: list[ScheduledLQRNode] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.nodes = self._build_schedule()

    def __call__(self, _time: float, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        scores = [self._schedule_distance(state, node.reference_state) for node in self.nodes]
        best_idx = int(np.argmin(scores))
        node = self.nodes[best_idx]

        delta = state - node.reference_state
        delta[1] = float(wrap_to_pi(delta[1]))
        delta[2] = float(wrap_to_pi(delta[2]))
        control = node.reference_input - float((node.gain.k @ delta.reshape(-1, 1)).item())
        return control

    def _build_schedule(self) -> list[ScheduledLQRNode]:
        schedule = []
        for alpha in np.linspace(0.0, 1.0, self.n_nodes):
            x_ref = self.target_position * alpha
            theta_ref = np.pi * (1.0 - alpha)
            reference_state = np.array([x_ref, theta_ref, theta_ref, 0.0, 0.0, 0.0], dtype=float)
            reference_input = self._best_feedforward_input(reference_state)
            a_mat, b_mat = linear_state_space(
                self.params,
                equilibrium_state=reference_state,
                equilibrium_input=reference_input,
            )
            try:
                gain = lqr_gain(a_mat, b_mat, self.q_mat, self.r_mat)
            except np.linalg.LinAlgError:
                continue
            schedule.append(
                ScheduledLQRNode(
                    reference_state=reference_state,
                    reference_input=reference_input,
                    gain=gain,
                )
            )
        if not schedule:
            raise RuntimeError("Failed to build any scheduled LQR nodes.")
        return schedule

    def _best_feedforward_input(self, reference_state: np.ndarray) -> float:
        def objective(cart_force: float) -> float:
            dx = cartpole_dynamics(0.0, reference_state, cart_force, self.params)
            return float(np.dot(dx[3:], dx[3:]))

        result = minimize_scalar(
            objective,
            bounds=(-self.params.force_limit, self.params.force_limit),
            method="bounded",
        )
        return float(result.x)

    def _schedule_distance(self, state: np.ndarray, reference_state: np.ndarray) -> float:
        angle_error_1 = float(wrap_to_pi(state[1] - reference_state[1]))
        angle_error_2 = float(wrap_to_pi(state[2] - reference_state[2]))
        velocity_error = state[3:] - reference_state[3:]
        return float(
            self.position_weight * (state[0] - reference_state[0]) ** 2
            + self.angle_weight * (angle_error_1**2 + angle_error_2**2)
            + self.velocity_weight * np.dot(velocity_error, velocity_error)
        )


def lqr_gain(a_mat: np.ndarray, b_mat: np.ndarray, q_mat: np.ndarray, r_mat: np.ndarray) -> LQRGain:
    s_mat = solve_continuous_are(a_mat, b_mat, q_mat, r_mat)
    k_mat = np.linalg.solve(r_mat, b_mat.T @ s_mat)
    eigs = np.linalg.eigvals(a_mat - b_mat @ k_mat)
    return LQRGain(k=k_mat, s=s_mat, closed_loop_eigs=eigs)
