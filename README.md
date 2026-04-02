# MAE589_AdvancedRobotics

Double inverted pendulum cart-pole 문제를 풀기 위한 Python 베이스라인입니다.

현재 포함된 내용:

- 비선형 동역학 모델
- LQR 제어
- sampling-based MPC
- MPPI 제어
- CasADi 기반 nonlinear trajectory optimization
- `gif` / `mp4` 저장 지원

프로젝트 구조:

- `src/double_inverted_pendulum/model.py`: 시스템 파라미터
- `src/double_inverted_pendulum/dynamics.py`: 비선형 동역학과 선형화
- `src/double_inverted_pendulum/environment.py`: 시뮬레이션 환경과 obstacle 슬롯
- `src/double_inverted_pendulum/kinematics.py`: cart, joint, end-tip 위치 계산
- `src/double_inverted_pendulum/controllers.py`: LQR 및 MPC controller
- `src/double_inverted_pendulum/simulation.py`: 폐루프 시뮬레이션
- `src/double_inverted_pendulum/visualization.py`: matplotlib 애니메이션과 저장
- `scripts/run_lqr_demo.py`: 빠른 실행 예제
- `scripts/run_mpc_demo.py`: sampling MPC 예제
- `scripts/run_mppi_demo.py`: MPPI 예제
- `scripts/run_casadi_trajopt_demo.py`: CasADi trajopt 예제

빠른 시작:

```bash
python -m pip install -e .
python scripts/run_lqr_demo.py
python scripts/run_mpc_demo.py
python scripts/run_mppi_demo.py
```

CasADi optional setup:

```bash
python -m pip install -e '.[casadi]'
```

CasADi nonlinear trajectory optimization demo:

```bash
python scripts/run_casadi_trajopt_demo.py
```

`casadi` extra는 nonlinear trajectory optimization을 위한 선택 설치입니다.

기본값은 optimized trajectory 자체를 시각화합니다. 실제 open-loop 재생을 보고 싶으면:

```bash
python scripts/run_casadi_trajopt_demo.py --replay-plan
```

모델 가정:

- 카트 질량 `M`
- 두 링크는 질량이 끝점에 집중된 point-mass pendulum
- 각도 `theta1`, `theta2`는 위쪽 직립 자세를 기준으로 측정
- `theta = 0`이 upright equilibrium
- 현재 control input은 카트에 가해지는 수평 force `u`로 두었습니다

남겨둔 실행 스크립트:

- `python scripts/run_lqr_demo.py`
- `python scripts/run_mpc_demo.py`
- `python scripts/run_mppi_demo.py`
- `python scripts/run_casadi_trajopt_demo.py`
