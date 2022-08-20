# gym ultrasound

- Ultrasound probe env and free space motion env with and without gripper
- Franka Emika Panda Robot model definitions for Mujoco.
- MuJoCo-based robot simulator.
- Python 3 API for controlling and monitoring the simulated Franka Emika Panda Robot.
- Low-level controllers: direct position and torque control.
- Higher-level controller: Force based variable impedance controller using torque controller

Robot models are in [gym_robotic_ultrasound/mujoco_panda/models](gym_robotic_ultrasound/mujoco_panda/models).

## Requirements

To use all functionalities of the provided library, the following dependencies have to be met.

- [mujoco_py](https://github.com/openai/mujoco-py)
- numpy (`pip install numpy`)
- scipy (`pip install scipy`)
- quaternion (`pip install numpy-quaternion`)
- tkinter (`apt-get install python3-tk) (only for visualised debugging`)

## Setup Instructions

Once mujoco_py is correctly installed, this library can be used by sourcing the `set_env.sh` file.

```bash
source set_env.sh
```

## Usage

