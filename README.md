# Optimal & Learning Control for Robotics

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)

> Personal coursework repository for **NYU ROB-GY 6323** — Reinforcement Learning and Optimal Control for Robotics, taught by [Ludovic Righetti](https://engineering.nyu.edu/faculty/ludovic-righetti).  
> Implements controllers of increasing complexity: from LQR to iterative LQR to Q-learning.

---

## Projects

### [Project 1 — Optimal Control of a 2D Quadrotor](Fall2022/Project1%20-%20Optimal%20Control/README.md)

Design controllers to make a planar quadrotor perform acrobatic maneuvers. Four parts of increasing complexity:

| Part | Method | Task |
|---|---|---|
| 1 | Setup | Discretise dynamics, derive hover control u* |
| 2 | Infinite-horizon LQR | Keep robot at rest under wind disturbances |
| 3 | Time-varying LQR | Track a circular trajectory |
| 4 | iLQR | Reach a vertical orientation; perform a full flip |

→ [View Project 1 README](Fall2022/Project1%20-%20Optimal%20Control/README.md)

---

### [Project 2 — Q-Learning: Inverted Pendulum](Fall2022/Project2%20-%20Reinforcement%20Learning/README.md)

Train a tabular Q-learning agent to swing a pendulum from rest to the inverted position. Two experiments compare torque limit configurations (±3 vs ±5).

| Part | Method | Task |
|---|---|---|
| 1 | Q-learning (table) | Learn a swing-up policy with ε-greedy exploration |
| 2 | Hyperparameter study | Compare controls `[-3,0,3]` vs `[-5,0,5]` |

→ [View Project 2 README](Fall2022/Project2%20-%20Reinforcement%20Learning/README.md)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Course

**NYU ROB-GY 6323** — Reinforcement Learning and Optimal Control for Robotics  
Instructor: [Ludovic Righetti](https://engineering.nyu.edu/faculty/ludovic-righetti), ECE-MAE, New York University
