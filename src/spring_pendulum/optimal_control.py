from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from double_inverted_pendulum.controllers import wrap_to_pi

from .dynamics import spring_cartpole_dynamics
from .model import SpringCartPoleParams, hanging_length


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
    hanging_radius: float,
) -> np.ndarray:
    initial_state = np.asarray(initial_state, dtype=float)
    goal_state = np.asarray(goal_state, dtype=float)
    tau = np.linspace(0.0, 1.0, horizon_steps + 1)
    smooth = 3.0 * tau**2 - 2.0 * tau**3

    ref = np.zeros((horizon_steps + 1, initial_state.size), dtype=float)
    ref[:, 0] = initial_state[0] + (goal_state[0] - initial_state[0]) * smooth
    ref[:, 1] = initial_state[1] + (goal_state[1] - initial_state[1]) * smooth
    radius_bump = np.sin(np.pi * tau) ** 2
    ref[:, 2] = initial_state[2] + (goal_state[2] - initial_state[2]) * smooth + 0.18 * (hanging_radius - goal_state[2]) * radius_bump
    ref[:, 3:] = initial_state[3:] + (goal_state[3:] - initial_state[3:]) * smooth[:, None]
    return ref


def optimize_trajectory_with_casadi(
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    params: SpringCartPoleParams,
    horizon_steps: int = 80,
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
    hanging_radius = hanging_length(params)
    ref_state = build_swing_reference(x0, x_goal, horizon_steps, hanging_radius)
    ref_control = np.zeros(horizon_steps, dtype=float)

    q_path = np.diag([4.0, 0.0, 3.0, 0.2, 0.15, 0.25])
    q_terminal = np.diag([55.0, 0.0, 35.0, 6.0, 6.0, 7.0])
    angle_path_weight = 16.0
    angle_terminal_weight = 220.0
    control_weight = 0.004
    control_smooth_weight = 0.0012

    opti = ca.Opti()
    x_var = opti.variable(6, horizon_steps + 1)
    u_var = opti.variable(1, horizon_steps)

    def dynamics_symbolic(state: ca.MX, cart_force: ca.MX) -> ca.MX:
        x_pos, theta, radius, x_dot, theta_dot, radius_dot = (
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            state[5],
        )
        cart_force = ca.fmax(-params.force_limit, ca.fmin(params.force_limit, cart_force))
        radius = ca.fmax(params.min_length, ca.fmin(params.max_length, radius))

        m0 = params.cart_mass
        m1 = params.bob_mass
        g = params.gravity

        mass_matrix = ca.vertcat(
            ca.horzcat(m0 + m1, m1 * radius * ca.cos(theta), m1 * ca.sin(theta)),
            ca.horzcat(m1 * radius * ca.cos(theta), m1 * radius**2, 0.0),
            ca.horzcat(m1 * ca.sin(theta), 0.0, m1),
        )
        generalized = ca.vertcat(
            cart_force
            - params.cart_damping * x_dot
            + m1 * radius * ca.sin(theta) * theta_dot**2
            - 2.0 * m1 * ca.cos(theta) * radius_dot * theta_dot,
            -2.0 * m1 * radius * radius_dot * theta_dot
            + m1 * g * radius * ca.sin(theta)
            - params.angular_damping * theta_dot,
            m1 * radius * theta_dot**2
            - params.spring_constant * (radius - params.rest_length)
            - m1 * g * ca.cos(theta)
            - params.radial_damping * radius_dot,
        )
        accelerations = ca.solve(mass_matrix, generalized)
        return ca.vertcat(x_dot, theta_dot, radius_dot, accelerations[0], accelerations[1], accelerations[2])

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
    opti.subject_to(opti.bounded(params.min_length, x_var[2, :], params.max_length))
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
        objective += control_weight * (u_k - ref_control[k]) ** 2
        objective += 3.5 * (x_k[2] - ref_state[k, 2]) ** 2
        objective -= 6.0 * ca.cos(x_k[1])
        if k > 0:
            objective += control_smooth_weight * (u_k - u_var[0, k - 1]) ** 2

    terminal_error = x_var[:, -1] - x_goal
    objective += ca.mtimes([terminal_error.T, q_terminal, terminal_error])
    objective += angle_terminal_weight * angular_cost(x_var[1, -1], x_goal[1])
    objective += 120.0 * (x_var[2, -1] - x_goal[2]) ** 2
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
        raise RuntimeError(
            "CasADi NLP solver backend not available. Install IPOPT or a CasADi build with sqpmethod support."
        )
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
    params: SpringCartPoleParams
    goal_state: np.ndarray
    dt: float = 0.05
    horizon: int = 30
    num_samples: int = 384
    noise_sigma: float = 7.5
    temperature: float = 6.0
    action_clip: float | None = None
    position_bounds: tuple[float, float] | None = None
    action_repeat: int = 1
    nominal_sequence: np.ndarray = field(default_factory=lambda: np.zeros(30, dtype=float))
    last_predicted_states: np.ndarray | None = field(default=None, init=False)
    last_sampled_states: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.goal_state = np.asarray(self.goal_state, dtype=float)
        if self.action_clip is None:
            self.action_clip = self.params.force_limit
        if self.nominal_sequence.shape[0] != self.horizon:
            self.nominal_sequence = np.zeros(self.horizon, dtype=float)
        self._rng = np.random.default_rng(21)
        self._hanging_length = hanging_length(self.params)

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
        x_error = states[:, :, 0] - self.goal_state[0]
        theta_error = wrap_to_pi(states[:, :, 1] - self.goal_state[1])
        radius_error = states[:, :, 2] - self.goal_state[2]
        x_dot = states[:, :, 3]
        theta_dot = states[:, :, 4]
        radius_dot = states[:, :, 5]

        uprightness = np.cos(theta_error)
        swing_progress = np.clip((uprightness + 1.0) / 2.0, 0.0, 1.0)

        cost = np.sum((0.12 + 3.2 * swing_progress) * x_error**2, axis=1)
        cost += np.sum(28.0 * theta_error**2, axis=1)
        cost += np.sum((1.6 + 4.8 * swing_progress) * radius_error**2, axis=1)
        cost += np.sum(0.10 * x_dot**2 + 0.22 * theta_dot**2 + 0.26 * radius_dot**2, axis=1)
        cost += np.sum(0.0015 * controls**2, axis=1)
        cost -= np.sum(15.0 * uprightness, axis=1)

        stretch_margin_low = states[:, :, 2] - self.params.min_length
        stretch_margin_high = self.params.max_length - states[:, :, 2]
        cost += np.sum(160.0 * np.maximum(0.0, 0.08 - stretch_margin_low) ** 2, axis=1)
        cost += np.sum(160.0 * np.maximum(0.0, 0.08 - stretch_margin_high) ** 2, axis=1)

        if self.position_bounds is not None:
            lower_margin = states[:, :, 0] - self.position_bounds[0]
            upper_margin = self.position_bounds[1] - states[:, :, 0]
            cost += np.sum(140.0 * np.maximum(0.0, 0.24 - lower_margin) ** 2, axis=1)
            cost += np.sum(140.0 * np.maximum(0.0, 0.24 - upper_margin) ** 2, axis=1)

        tail = rollout[:, -min(8, rollout.shape[1] - 1) :, :]
        tail_theta = wrap_to_pi(tail[:, :, 1] - self.goal_state[1])
        tail_radius = tail[:, :, 2] - self.goal_state[2]
        cost += np.sum(45.0 * (tail[:, :, 0] - self.goal_state[0]) ** 2, axis=1)
        cost += np.sum(120.0 * tail_theta**2, axis=1)
        cost += np.sum(32.0 * tail_radius**2, axis=1)
        cost += np.sum(4.0 * tail[:, :, 3] ** 2 + 5.5 * tail[:, :, 4] ** 2 + 4.0 * tail[:, :, 5] ** 2, axis=1)

        terminal = rollout[:, -1, :]
        terminal_theta = wrap_to_pi(terminal[:, 1] - self.goal_state[1])
        terminal_radius = terminal[:, 2] - self.goal_state[2]
        cost += 250.0 * (terminal[:, 0] - self.goal_state[0]) ** 2
        cost += 700.0 * terminal_theta**2
        cost += 220.0 * terminal_radius**2
        cost += 10.0 * terminal[:, 3] ** 2 + 16.0 * terminal[:, 4] ** 2 + 10.0 * terminal[:, 5] ** 2

        return cost.astype(float)

    def _predict_states(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float).copy()
        rollout = np.zeros((controls.shape[0] + 1, x.size), dtype=float)
        rollout[0] = x
        for idx, u in enumerate(controls, start=1):
            x = self._rk4_step(x, float(u))
            if self.position_bounds is not None:
                x[0] = np.clip(x[0], self.position_bounds[0], self.position_bounds[1])
            x[2] = np.clip(x[2], self.params.min_length, self.params.max_length)
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
            x[:, 2] = np.clip(x[:, 2], self.params.min_length, self.params.max_length)
            rollout[:, idx + 1, :] = x
        return rollout

    def _rk4_step(self, state: np.ndarray, control: float) -> np.ndarray:
        k1 = spring_cartpole_dynamics(0.0, state, control, self.params)
        k2 = spring_cartpole_dynamics(0.0, state + 0.5 * self.dt * k1, control, self.params)
        k3 = spring_cartpole_dynamics(0.0, state + 0.5 * self.dt * k2, control, self.params)
        k4 = spring_cartpole_dynamics(0.0, state + self.dt * k3, control, self.params)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_step_batch(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        k1 = self._dynamics_batch(state, control)
        k2 = self._dynamics_batch(state + 0.5 * self.dt * k1, control)
        k3 = self._dynamics_batch(state + 0.5 * self.dt * k2, control)
        k4 = self._dynamics_batch(state + self.dt * k3, control)
        return state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _dynamics_batch(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        theta = state[:, 1]
        radius = np.clip(state[:, 2], self.params.min_length, self.params.max_length)
        x_dot = state[:, 3]
        theta_dot = state[:, 4]
        radius_dot = state[:, 5]
        clipped = np.clip(control, -self.params.force_limit, self.params.force_limit)

        m0 = self.params.cart_mass
        m1 = self.params.bob_mass
        g = self.params.gravity

        mass_matrix = np.zeros((state.shape[0], 3, 3), dtype=float)
        mass_matrix[:, 0, 0] = m0 + m1
        mass_matrix[:, 0, 1] = m1 * radius * np.cos(theta)
        mass_matrix[:, 0, 2] = m1 * np.sin(theta)
        mass_matrix[:, 1, 0] = mass_matrix[:, 0, 1]
        mass_matrix[:, 1, 1] = m1 * radius**2
        mass_matrix[:, 2, 0] = mass_matrix[:, 0, 2]
        mass_matrix[:, 2, 2] = m1

        generalized = np.zeros((state.shape[0], 3), dtype=float)
        generalized[:, 0] = (
            clipped
            - self.params.cart_damping * x_dot
            + m1 * radius * np.sin(theta) * theta_dot**2
            - 2.0 * m1 * np.cos(theta) * radius_dot * theta_dot
        )
        generalized[:, 1] = (
            -2.0 * m1 * radius * radius_dot * theta_dot
            + m1 * g * radius * np.sin(theta)
            - self.params.angular_damping * theta_dot
        )
        generalized[:, 2] = (
            m1 * radius * theta_dot**2
            - self.params.spring_constant * (radius - self.params.rest_length)
            - m1 * g * np.cos(theta)
            - self.params.radial_damping * radius_dot
        )
        accelerations = np.linalg.solve(mass_matrix, generalized[..., None])[..., 0]
        return np.column_stack((x_dot, theta_dot, radius_dot, accelerations))
