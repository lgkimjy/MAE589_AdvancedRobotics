# MAE589_AdvancedRobotics

Python examples for single and double inverted pendulum cart-pole systems, plus a spring pendulum cart-pole case.

## Install

Base install:

```bash
python -m pip install -e .
```

Install with the optional CasADi dependency:

```bash
python -m pip install -e '.[casadi]'
```

## Layout

- `src/double_inverted_pendulum/`: double inverted pendulum models, controllers, simulation, and visualization
- `src/single_inverted_pendulum/`: single inverted pendulum models, controllers, simulation, and visualization
- `src/spring_pendulum/`: spring cart-pole dynamics, simulation, and visualization
- `scripts/`: runnable demos
- `outputs/`: saved animation outputs

## Scripts

- `scripts/run_lqr_demo.py`: double pendulum LQR demo
- `scripts/run_mpc_demo.py`: double pendulum sampling MPC demo
- `scripts/run_mppi_demo.py`: double pendulum MPPI demo
- `scripts/run_chaos_demo.py`: double pendulum zero-input passive motion demo
- `scripts/run_casadi_trajopt_demo.py`: CasADi trajectory optimization demo
- `scripts/run_single_mppi_demo.py`: single pendulum MPPI demo
- `scripts/run_spring_casadi_trajopt_demo.py`: spring pendulum cart-pole CasADi trajectory optimization demo
- `scripts/run_spring_mppi_demo.py`: spring pendulum cart-pole MPPI demo
- `scripts/run_spring_pendulum_demo.py`: spring pendulum cart-pole passive demo

## Model

- The state contains cart position, two link angles, and their velocities.
- `theta1` and `theta2` are measured from the upright configuration, so `0` is the upright equilibrium.
- The control input is the horizontal cart force `u`.
- In the double pendulum model, each link mass is concentrated at the tip as a point mass.
- In the spring pendulum cart-pole model, the state contains cart position, pendulum angle, spring length, and their rates.
