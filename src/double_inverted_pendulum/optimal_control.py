from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from .controllers import wrap_to_pi
from .dynamics import cartpole_dynamics
from .model import CartPoleParams


@dataclass(frozen=True)
class TrajectoryPlan:
    time: np.ndarray
    state: np.ndarray
    control: np.ndarray
    reference_state: np.ndarray
    reference_control: np.ndarray


def build_swing_reference(
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    initial_state = np.asarray(initial_state, dtype=float)
    goal_state = np.asarray(goal_state, dtype=float)
    tau = np.linspace(0.0, 1.0, horizon_steps + 1)
    smooth = 3.0 * tau**2 - 2.0 * tau**3

    ref = np.zeros((horizon_steps + 1, initial_state.size), dtype=float)
    ref[:, 0] = initial_state[0] + (goal_state[0] - initial_state[0]) * smooth
    ref[:, 1] = initial_state[1] + (goal_state[1] - initial_state[1]) * smooth
    ref[:, 2] = initial_state[2] + (goal_state[2] - initial_state[2]) * smooth
    return ref


def compute_feedforward_input(reference_state: np.ndarray, params: CartPoleParams) -> float:
    reference_state = np.asarray(reference_state, dtype=float)

    def objective(cart_force: float) -> float:
        dx = cartpole_dynamics(0.0, reference_state, cart_force, params)
        return float(np.dot(dx[3:], dx[3:]))

    result = minimize_scalar(
        objective,
        bounds=(-params.force_limit, params.force_limit),
        method="bounded",
    )
    return float(result.x)


def optimize_trajectory_with_casadi(
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    params: CartPoleParams,
    horizon_steps: int = 120,
    dt: float = 0.05,
    position_bounds: tuple[float, float] | None = None,
    max_iter: int = 2000,
) -> TrajectoryPlan:
    try:
        import casadi as ca
    except ImportError as exc:
        raise ImportError("casadi is required for nonlinear trajectory optimization.") from exc

    x0 = np.asarray(initial_state, dtype=float)
    x_goal = np.asarray(goal_state, dtype=float)
    ref_state = build_swing_reference(x0, x_goal, horizon_steps)
    ref_control = np.array([compute_feedforward_input(ref_state[k], params) for k in range(horizon_steps)], dtype=float)

    q_path = np.diag([8.0, 0.0, 0.0, 0.35, 0.2, 0.2])
    q_terminal = np.diag([65.0, 0.0, 0.0, 5.0, 3.0, 3.0])
    angle_path_weight = 10.0
    angle_terminal_weight = 140.0
    control_weight = 0.003
    control_smooth_weight = 0.0008

    opti = ca.Opti()
    x_var = opti.variable(6, horizon_steps + 1)
    u_var = opti.variable(1, horizon_steps)

    def dynamics_symbolic(state: ca.MX, cart_force: ca.MX) -> ca.MX:
        _x_pos, theta1, theta2, x_dot, theta1_dot, theta2_dot = state[0], state[1], state[2], state[3], state[4], state[5]
        cart_force = ca.fmax(-params.force_limit, ca.fmin(params.force_limit, cart_force))

        m0 = params.cart_mass
        m1 = params.link1_mass
        m2 = params.link2_mass
        l1 = params.link1_length
        l2 = params.link2_length
        g = params.gravity

        mass_matrix = ca.vertcat(
            ca.horzcat(m0 + m1 + m2, (m1 + m2) * l1 * ca.cos(theta1), m2 * l2 * ca.cos(theta2)),
            ca.horzcat((m1 + m2) * l1 * ca.cos(theta1), (m1 + m2) * l1**2, m2 * l1 * l2 * ca.cos(theta1 - theta2)),
            ca.horzcat(m2 * l2 * ca.cos(theta2), m2 * l1 * l2 * ca.cos(theta1 - theta2), m2 * l2**2),
        )
        generalized_forces = ca.vertcat(
            cart_force
            - params.cart_damping * x_dot
            + (m1 + m2) * l1 * ca.sin(theta1) * theta1_dot**2
            + m2 * l2 * ca.sin(theta2) * theta2_dot**2,
            -params.joint1_damping * theta1_dot
            - m2 * l1 * l2 * ca.sin(theta1 - theta2) * theta2_dot**2
            + (m1 + m2) * g * l1 * ca.sin(theta1),
            -params.joint2_damping * theta2_dot
            + m2 * l1 * l2 * ca.sin(theta1 - theta2) * theta1_dot**2
            + m2 * g * l2 * ca.sin(theta2),
        )
        accelerations = ca.solve(mass_matrix, generalized_forces)
        return ca.vertcat(x_dot, theta1_dot, theta2_dot, accelerations[0], accelerations[1], accelerations[2])

    def rk4_step(state: ca.MX, cart_force: ca.MX) -> ca.MX:
        k1 = dynamics_symbolic(state, cart_force)
        k2 = dynamics_symbolic(state + 0.5 * dt * k1, cart_force)
        k3 = dynamics_symbolic(state + 0.5 * dt * k2, cart_force)
        k4 = dynamics_symbolic(state + dt * k3, cart_force)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def angular_cost(theta: ca.MX, theta_ref: float) -> ca.MX:
        return 1.0 - ca.cos(theta - theta_ref)

    opti.subject_to(x_var[:, 0] == x0)
    opti.subject_to(opti.bounded(-params.force_limit, u_var, params.force_limit))
    if position_bounds is not None:
        opti.subject_to(opti.bounded(position_bounds[0], x_var[0, :], position_bounds[1]))

    objective = 0
    for k in range(horizon_steps):
        x_k = x_var[:, k]
        u_k = u_var[0, k]
        x_next = rk4_step(x_k, u_k)
        opti.subject_to(x_var[:, k + 1] == x_next)

        dx = x_k - ref_state[k]
        objective += ca.mtimes([dx.T, q_path, dx])
        objective += angle_path_weight * angular_cost(x_k[1], ref_state[k, 1])
        objective += angle_path_weight * angular_cost(x_k[2], ref_state[k, 2])
        objective += control_weight * (u_k - ref_control[k]) ** 2
        if k > 0:
            objective += control_smooth_weight * (u_k - u_var[0, k - 1]) ** 2

    terminal_error = x_var[:, -1] - x_goal
    objective += ca.mtimes([terminal_error.T, q_terminal, terminal_error])
    objective += angle_terminal_weight * angular_cost(x_var[1, -1], x_goal[1])
    objective += angle_terminal_weight * angular_cost(x_var[2, -1], x_goal[2])
    opti.minimize(objective)

    opti.set_initial(x_var, ref_state.T)
    opti.set_initial(u_var, ref_control.reshape(1, -1))

    if ca.has_nlpsol("ipopt"):
        solver_name = "ipopt"
        solver_opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": max_iter,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 8,
            "print_time": False,
        }
    elif ca.has_nlpsol("sqpmethod") and ca.has_conic("qrqp"):
        solver_name = "sqpmethod"
        solver_opts = {
            "qpsol": "qrqp",
            "print_time": False,
            "print_header": False,
            "print_iteration": False,
            "print_status": False,
            "max_iter": min(max_iter, 250),
            "tol_pr": 1e-4,
            "tol_du": 1e-4,
            "qpsol_options": {
                "print_iter": False,
                "print_header": False,
                "error_on_fail": False,
            },
        }
    else:
        raise RuntimeError("CasADi NLP solver backend not available. Install IPOPT or a CasADi build with sqpmethod support.")
    opti.solver(solver_name, solver_opts)

    try:
        solution = opti.solve()
    except RuntimeError as exc:
        if solver_name == "sqpmethod":
            debug_state = np.asarray(opti.debug.value(x_var), dtype=float).T
            debug_control = np.asarray(opti.debug.value(u_var), dtype=float).reshape(-1)
            if np.all(np.isfinite(debug_state)) and np.all(np.isfinite(debug_control)):
                time = np.arange(horizon_steps + 1, dtype=float) * dt
                ref_control_full = np.concatenate([ref_control, ref_control[-1:]])
                return TrajectoryPlan(
                    time=time,
                    state=debug_state,
                    control=np.concatenate([debug_control, debug_control[-1:]]),
                    reference_state=ref_state,
                    reference_control=ref_control_full,
                )
        raise RuntimeError("CasADi trajectory optimization failed to converge.") from exc

    state_sol = np.asarray(solution.value(x_var), dtype=float).T
    control_sol = np.asarray(solution.value(u_var), dtype=float).reshape(-1)
    time = np.arange(horizon_steps + 1, dtype=float) * dt
    ref_control_full = np.concatenate([ref_control, ref_control[-1:]])
    return TrajectoryPlan(
        time=time,
        state=state_sol,
        control=np.concatenate([control_sol, control_sol[-1:]]),
        reference_state=ref_state,
        reference_control=ref_control_full,
    )
@dataclass
class MPPIController:
    params: CartPoleParams
    goal_state: np.ndarray
    reference_time: np.ndarray | None = None
    reference_state: np.ndarray | None = None
    reference_control: np.ndarray | None = None
    dt: float = 0.05
    horizon: int = 28
    num_samples: int = 192
    noise_sigma: float = 11.0
    temperature: float = 6.0
    action_clip: float | None = None
    position_bounds: tuple[float, float] | None = None
    action_repeat: int = 2
    goal_ramp_time: float = 8.0
    nominal_sequence: np.ndarray = field(default_factory=lambda: np.zeros(28, dtype=float))
    last_predicted_states: np.ndarray | None = field(default=None, init=False)
    last_sampled_states: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.goal_state = np.asarray(self.goal_state, dtype=float)
        if self.reference_time is not None:
            self.reference_time = np.asarray(self.reference_time, dtype=float)
        if self.reference_state is not None:
            self.reference_state = np.asarray(self.reference_state, dtype=float)
        if self.reference_control is not None:
            self.reference_control = np.asarray(self.reference_control, dtype=float)
        if self.action_clip is None:
            self.action_clip = self.params.force_limit
        if self.nominal_sequence.shape[0] != self.horizon:
            self.nominal_sequence = np.zeros(self.horizon, dtype=float)
        self._rng = np.random.default_rng(18)

    def __call__(self, time: float, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=float)
        coarse_horizon = int(np.ceil(self.horizon / max(1, self.action_repeat)))
        coarse_nominal = self.nominal_sequence[:: max(1, self.action_repeat)][:coarse_horizon]
        noise = self._rng.normal(0.0, self.noise_sigma, size=(self.num_samples, coarse_horizon))
        candidate_coarse = np.clip(coarse_nominal[None, :] + noise, -self.action_clip, self.action_clip)
        candidate_u = np.repeat(candidate_coarse, max(1, self.action_repeat), axis=1)[:, : self.horizon]
        sampled_states = self._predict_states_batch(state, candidate_u)
        ref_states, ref_controls = self._reference_rollout(float(time), state)
        costs = self._trajectory_cost_batch(sampled_states, candidate_u, ref_states, ref_controls)
        beta = float(np.min(costs))
        weights = np.exp(-(costs - beta) / max(self.temperature, 1e-6))
        weights /= np.sum(weights)
        self.nominal_sequence = np.sum(weights[:, None] * candidate_u, axis=0)
        self.last_predicted_states = self._predict_states(state, self.nominal_sequence)
        sample_stride = max(1, self.num_samples // 48)
        self.last_sampled_states = sampled_states[::sample_stride]
        action = float(np.clip(self.nominal_sequence[0], -self.action_clip, self.action_clip))
        self.nominal_sequence[:-1] = self.nominal_sequence[1:]
        self.nominal_sequence[-1] = 0.0
        return action

    def _trajectory_cost_batch(
        self,
        rollout: np.ndarray,
        controls: np.ndarray,
        reference_states: np.ndarray,
        reference_controls: np.ndarray,
    ) -> np.ndarray:
        states = rollout[:, 1:, :]
        ref_states = reference_states[1:, :]
        angle1 = wrap_to_pi(states[:, :, 1] - ref_states[None, :, 1])
        angle2 = wrap_to_pi(states[:, :, 2] - ref_states[None, :, 2])
        x_error = states[:, :, 0] - ref_states[None, :, 0]
        x_goal_error = states[:, :, 0] - self.goal_state[0]
        rates = states[:, :, 3:] - ref_states[None, :, 3:]
        uprightness = np.cos(states[:, :, 1]) + np.cos(states[:, :, 2])
        progress = np.clip((uprightness + 2.0) / 4.0, 0.0, 1.0)
        ramp = np.clip(reference_states[1:, 0] - reference_states[0, 0], 0.0, abs(self.goal_state[0] - reference_states[0, 0]))
        ramp = ramp / max(abs(self.goal_state[0] - reference_states[0, 0]), 1e-6)

        cost = np.sum((0.10 + 2.4 * ramp[None, :] + 1.8 * progress) * x_error**2, axis=1)
        cost += np.sum(24.0 * angle1**2 + 24.0 * angle2**2, axis=1)
        cost += np.sum(0.18 * rates[:, :, 0] ** 2 + 0.14 * rates[:, :, 1] ** 2 + 0.14 * rates[:, :, 2] ** 2, axis=1)
        cost += np.sum(0.0022 * (controls - reference_controls[None, :]) ** 2, axis=1)
        cost -= np.sum(10.0 * uprightness + 3.0 * progress * np.cos(angle1 - angle2), axis=1)
        cost += np.sum(2.2 * progress * x_goal_error**2, axis=1)

        if self.position_bounds is not None:
            lower_margin = states[:, :, 0] - self.position_bounds[0]
            upper_margin = self.position_bounds[1] - states[:, :, 0]
            cost += np.sum(160.0 * np.maximum(0.0, 0.28 - lower_margin) ** 2, axis=1)
            cost += np.sum(160.0 * np.maximum(0.0, 0.28 - upper_margin) ** 2, axis=1)

        terminal = rollout[:, -1, :]
        terminal_ref = reference_states[-1]
        terminal_angle1 = wrap_to_pi(terminal[:, 1] - terminal_ref[1])
        terminal_angle2 = wrap_to_pi(terminal[:, 2] - terminal_ref[2])
        terminal_rates = terminal[:, 3:] - terminal_ref[3:]
        tail = rollout[:, -min(6, rollout.shape[1] - 1) :, :]
        tail_angle1 = wrap_to_pi(tail[:, :, 1] - self.goal_state[1])
        tail_angle2 = wrap_to_pi(tail[:, :, 2] - self.goal_state[2])
        tail_rates = tail[:, :, 3:]
        cost += np.sum(30.0 * (tail[:, :, 0] - self.goal_state[0]) ** 2, axis=1)
        cost += np.sum(90.0 * tail_angle1**2 + 90.0 * tail_angle2**2, axis=1)
        cost += np.sum(4.0 * tail_rates[:, :, 0] ** 2 + 3.0 * tail_rates[:, :, 1] ** 2 + 3.0 * tail_rates[:, :, 2] ** 2, axis=1)
        cost += 180.0 * (terminal[:, 0] - terminal_ref[0]) ** 2
        cost += 420.0 * terminal_angle1**2 + 420.0 * terminal_angle2**2
        cost += 18.0 * terminal_rates[:, 0] ** 2 + 12.0 * terminal_rates[:, 1] ** 2 + 12.0 * terminal_rates[:, 2] ** 2
        cost += 120.0 * (terminal[:, 0] - self.goal_state[0]) ** 2
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
        theta1 = state[:, 1]
        theta2 = state[:, 2]
        x_dot = state[:, 3]
        theta1_dot = state[:, 4]
        theta2_dot = state[:, 5]
        clipped = np.clip(control, -self.params.force_limit, self.params.force_limit)

        m0 = self.params.cart_mass
        m1 = self.params.link1_mass
        m2 = self.params.link2_mass
        l1 = self.params.link1_length
        l2 = self.params.link2_length
        g = self.params.gravity

        mass_matrix = np.empty((state.shape[0], 3, 3), dtype=float)
        mass_matrix[:, 0, 0] = m0 + m1 + m2
        mass_matrix[:, 0, 1] = (m1 + m2) * l1 * np.cos(theta1)
        mass_matrix[:, 0, 2] = m2 * l2 * np.cos(theta2)
        mass_matrix[:, 1, 0] = mass_matrix[:, 0, 1]
        mass_matrix[:, 1, 1] = (m1 + m2) * l1**2
        mass_matrix[:, 1, 2] = m2 * l1 * l2 * np.cos(theta1 - theta2)
        mass_matrix[:, 2, 0] = mass_matrix[:, 0, 2]
        mass_matrix[:, 2, 1] = mass_matrix[:, 1, 2]
        mass_matrix[:, 2, 2] = m2 * l2**2

        generalized = np.empty((state.shape[0], 3), dtype=float)
        generalized[:, 0] = (
            clipped
            - self.params.cart_damping * x_dot
            + (m1 + m2) * l1 * np.sin(theta1) * theta1_dot**2
            + m2 * l2 * np.sin(theta2) * theta2_dot**2
        )
        generalized[:, 1] = (
            -self.params.joint1_damping * theta1_dot
            - m2 * l1 * l2 * np.sin(theta1 - theta2) * theta2_dot**2
            + (m1 + m2) * g * l1 * np.sin(theta1)
        )
        generalized[:, 2] = (
            -self.params.joint2_damping * theta2_dot
            + m2 * l1 * l2 * np.sin(theta1 - theta2) * theta1_dot**2
            + m2 * g * l2 * np.sin(theta2)
        )

        accelerations = np.linalg.solve(mass_matrix, generalized[..., None])[..., 0]
        return np.column_stack((x_dot, theta1_dot, theta2_dot, accelerations))

    def _reference_rollout(self, current_time: float, current_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        horizon_times = current_time + self.dt * np.arange(self.horizon + 1, dtype=float)
        if self.reference_time is not None and self.reference_state is not None:
            ref_state = np.column_stack(
                [
                    np.interp(horizon_times, self.reference_time, self.reference_state[:, idx])
                    for idx in range(self.reference_state.shape[1])
                ]
            )
            if self.reference_control is not None:
                control_times = self.reference_time[: self.reference_control.shape[0]]
                ref_control = np.interp(horizon_times[:-1], control_times, self.reference_control)
            else:
                ref_control = np.zeros(self.horizon, dtype=float)
            return ref_state, ref_control

        ramp = np.clip(horizon_times / max(self.goal_ramp_time, self.dt), 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        ref_state = np.repeat(np.asarray(current_state, dtype=float)[None, :], self.horizon + 1, axis=0)
        ref_state[:, 0] = current_state[0] + (self.goal_state[0] - current_state[0]) * ramp
        ref_state[:, 1] = np.pi + (self.goal_state[1] - np.pi) * ramp
        ref_state[:, 2] = np.pi + (self.goal_state[2] - np.pi) * ramp
        ref_state[:, 3:] = 0.0
        return ref_state, np.zeros(self.horizon, dtype=float)
