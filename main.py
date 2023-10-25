import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    # Load the scenario and create the environment
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        done_callback=scenario.is_done,
        shared_viewer=True
    )

    # Reset the environment
    env.reset()

    step = 0
    max_step = 400

    while True:
        # Take a predefined action: [noop, move right, move left, move up, move down]
        action_n = np.array([0, 1, 0, 0, 1])

        # Step through the environment with the predefined action
        next_obs_n, reward_n, done_n, _ = env.step(action_n, 1)

        # Render the environment to obtain an image
        image = env.render("rgb_array")[0]

        step += 1

        if True in done_n or step > max_step:
            break
