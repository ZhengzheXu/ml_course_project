# Machine Learning Course Project
This is a course project for the course **Introduction to Machine Learning** (2023 Fall) at Harbin Institute of Technology (Shenzhen).

The project is about the chase-escape game. The goal is to train the evader to escape from the hunter while avoiding obstacles. 

In this project, we acomplish the task by two methods:
+ **Model Predictive Control (MPC)**: 
We formulate a model predictive control (MPC) problem for the chase-escape game. The MPC problem is formulated using CasADi and solved by IPOPT in real-time.
+ **Reinforcement Learning (RL)**:
We use PPO (Proximal Policy Optimization) to train the evader to escape from the hunter. The network is an actor-critic network. 

## 0. Project Structure
+ `mpc_main.py` is a model predictive control (MPC) algorithm for the chase-escape game. It is a baseline for the project *(some parameters can be tuned for better performance)*. Maybe you can try to adopt a CBF (control barrier function) to get better obstacle avoidance performance.
+ `ppo_main.py` can use to train and evaluate the evader. We provide a trained model in `saved_models/My_Tag_Model`. Feel free to use it to train your own model.

## 1. Environment Configuration

**Particle Ball Chase-Escape Environment**

> Reference [Blog](https://blog.csdn.net/kysguqfxfr/article/details/100070584?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

## 2. Installation

**System:** Ubuntu 18.04/20.04

**Dependencies:**
- gymnasium
- pyglet
- CasADi (for `mpc_main.py`)

> Run `demo.py` to check the environment, and install any missing packages.

## 3. Environment Introduction

> Using `demo.py` as an example.

First, create and initialize the environment:

```python
scenario = scenarios.load("simple_tag.py").Scenario()
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.is_done, shared_viewer=True)

env.reset()
image = env.render("rgb_array")  # Use this method to read the image
```

The action is a five-dimensional vector with ranges `[0,1]`, controlling the direction and magnitude of acceleration:

```python
# [noop, move right, move left, move up, move down]
act_n = np.array([0, 1, 0, 0, 1])
```

The state is an RGB image of size `(800,800,3)`, suitable for object detection to obtain position information or for end-to-end training of agents. The actual scale of the entire scene is [-1, 1] * [-1, 1], with the center of the screen as the origin. The image size is 800 * 800 pixels:

```python
image = env.render("rgb_array")  # Use this method to read the image
```

In addition to image data, you can also directly obtain the state using methods common to general environments:

```python
obs_n = env.reset()
.
.
.
next_obs_n, reward_n, done_n, _ = env.step(act_n, 1)
```

In the environment class in the simple_tag.py file, the observation() function contains code to directly obtain position information, among other things.

However, this approach does not comply with project's requirements, as the intention is for students to use knowledge of image processing to obtain state information.

Obstacle positions: `[-0.35, 0.35], [0.35, 0.35], [0, -0.35]`, with a radius of 0.05.

Reset function, where the first position is the hunter, and the second position is the controlled agent:

```python
env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]))
```

The episode ends when an obstacle (excluding the boundary) is hit or when the hunter catches up with the evader:

```python
if True in done_n:
    break
```

## 4. Environment Files
The most important environment files are core.py, simple_tag.py, and environment.py. The rough call relationship among these three files is as follows: core is called by simple_tag, simple_tag is called by environment.

- In the core.py file, various entities in the environment are mainly declared: agent, border, landmark, check, etc.
- The simple_tag.py file mainly sets parameters for entities (initial positions, accelerations, etc.) and reward settings.
- The environment.py file is the classic environment interface in reinforcement learning algorithms (step, reset, reward, etc.). You can rewrite the self._set_action function in the given task according to your needs.
