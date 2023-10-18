import numpy as np

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import matplotlib.pyplot as plt 
import time

if __name__ == '__main__':
    # parse arguments
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.is_done, shared_viewer = True)

    # env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]))
    env.reset()
    image = env.render("rgb_array")  # read image
    step = 0
    total_step = 0
    max_step = 400
    # file = open('./course/label.txt', 'w')
    while True:
        # [noop, move right, move left, move up, move down]
        act_n = np.array([0,1,0,0,1])
        next_obs_n, reward_n, done_n, _ = env.step(act_n, 1)
        # print(f"vel: {np.linalg.norm(next_obs_n[0][-4:-2])}, vel2: {np.linalg.norm(next_obs_n[0][-2:])}")
        image = env.render("rgb_array")[0]  # read image
        
        # print(f"shape: {np.shape(image)}")
        step += 1
        if step % 50:
            # print(image)
            # plt.imshow(image)
            # plt.show()
            time.sleep(0.0167) # 60 fps
            # plt.close()

        if True in done_n or step > max_step:
            break
