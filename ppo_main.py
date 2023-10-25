import os
import sys
import datetime
import math

import torch
import numpy as np
import cv2

from agents.PPO.agent import PPO

import gymnasium as gym

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from ppo_function import PPOConfig, env_agent_config, train, eval

# Additional code for setting paths
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # Add current terminal path to sys.path

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Obtain current time

if __name__ == "__main__":
    # Initialize PPO configuration
    cfg = PPOConfig()

    # Create environment and agent
    env, agent = env_agent_config(cfg)

    if cfg.train:
        # Train the agent
        if cfg.load_model:
            print(">>>>>>>>>> Load model <<<<<<<<<<<<<")
            agent.load(path=cfg.model_path, i_ep=1016)
        rewards, ma_rewards = train(cfg, env, agent)
    else:
        # Evaluate the agent
        print(">>>>>>>>>> Load model <<<<<<<<<<<<<")
        epoch = 0
        ss_epoch = 0
        i_step_all = 0
        for i in range(7420, 7471, 10):
            print(i)
            epoch += 1
            agent.load(path=cfg.model_path, i_ep=i)
            rewards, ma_rewards, i_step, done = eval(cfg, env, agent)
            if done:
                i_step_all += i_step
                ss_epoch += 1

        success_rate = ss_epoch / epoch if epoch > 0 else 0
        average_steps = i_step_all / ss_epoch if ss_epoch > 0 else 0

        print("Success Rate:", success_rate)
        print("Average Steps:", average_steps)