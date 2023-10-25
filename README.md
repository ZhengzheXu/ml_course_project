# Machine Learning Course Project
Course project for machine learning at HITSZ

## 1. Environment Configuration

**Particle Ball Chase-Escape Environment**

> Reference [Blog](https://blog.csdn.net/kysguqfxfr/article/details/100070584?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

## 2. Installation

**System:** Ubuntu 18.04/20.04

**Dependencies:**
- gymnasium
- pyglet

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

## 5. Task Requirements
Avoid being caught by the hunter and reach the checkpoint ([-0.5, -0.5]). 