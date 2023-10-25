#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Additional code for setting paths
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # Add current terminal path to sys.path

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Obtain current time

class PPOConfig:
    def __init__(self) -> None:
        # Algorithm configuration
        self.algo = 'PPO'

        # Environment configuration
        self.env = 'MULTIAGENT-ENVS'

        # Random seed for reproducibility
        self.seed = 15

        # Display configuration
        self.show_image = False  # Set to True to render images

        # Model and training configuration
        self.load_model = False  # Set to True to load a pre-trained model
        self.train = False  # Set to True to train the model

        # Path to save trained models
        self.model_path = 'saved_models/My_Tag_Model/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Replay buffer configuration
        self.capacity = int(2e5)  # Replay buffer size

        # Mini-batch size for training
        self.batch_size = 512

        # Discount factor for future rewards
        self.gamma = 0.99

        # Learning rate for the optimizer
        self.lr = 1e-4

        # PPO clipping parameter
        self.eps_clip = 0.2

        # Number of optimization epochs
        self.K_epochs = 80

        # Frequency of learning updates
        self.update_every = 1  # Learn every UPDATE_EVERY time steps.

        # Training episodes and steps
        self.train_eps = 10000
        self.train_steps = 200

        # Evaluation episodes and steps
        self.eval_eps = 1
        self.eval_steps = 2000

        # Exploration-exploitation parameters
        self.eps_start = 1.0
        self.eps_decay = 0.995
        self.eps_end = 0.01

        # Device for computation (CPU in this case)
        self.device = torch.device("cpu")

        # Number of frames for the environment
        self.frames = 10

def env_agent_config(cfg: PPOConfig):
    """
    Create environment and agent.
    
    Args:
        cfg (PPOConfig): Configuration object.

    Returns:
        MultiAgentEnv: The environment.
        PPO: The agent.
    """
    # Load the scenario and create the world
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()

    # Create the environment with MultiAgentEnv
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        done_callback=scenario.is_done,
        shared_viewer=True
    )

    # Define action and state dimensions
    action_dim = 2
    state_len = 1
    state_dim = 12 * state_len
    
    print(f"Action dim: {action_dim}, State dim: {state_dim}")

    # Create the PPO agent
    N = cfg.frames
    agent = PPO(state_dim * N, action_dim, cfg)

    return env, agent

def normalize(state, observation_space=None):
    """Normalize the input state vector."""
    return state / np.linalg.norm(state)

def stack_frame(stacked_frames, frame, is_new, cfg: PPOConfig):
    """
    Stack frames for temporal information.

    Args:
        stacked_frames (List): List of stacked frames.
        frame: New frame to be added.
        is_new (bool): Flag indicating whether the frame is new.
        cfg (PPOConfig): Configuration object.

    Returns:
        List: Updated list of stacked frames.
    """
    if is_new:
        stacked_frames.extend([frame] * cfg.frames)
    else:
        stacked_frames[:-1] = stacked_frames[1:]
        stacked_frames[-1] = frame

    return stacked_frames

def train(cfg: PPOConfig, env, agent):
    """
    Train the model, save the agent and results during training.
    
    Args:
        cfg (PPOConfig): Configuration object.
        env: The environment.
        agent: The agent.

    Returns:
        List: Rewards.
        List: Moving average rewards.
    """
    print('Start training!')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    
    rewards = []
    ma_rewards = []  # moving average reward

    eps = cfg.eps_start
    for i_ep in range(cfg.train_eps):
        observation, info = env.reset()

        # Preprocess the initial observation
        image = env.render("rgb_array")[0]
        obs_img = img2obs(image)
        stacked_frames = [obs_img] * cfg.frames
        stacked_frames = stack_frame(stacked_frames, obs_img, False, cfg)
        state = normalize(np.concatenate(stacked_frames, axis=0))
        
        ep_reward = 0
        i_step = 0

        while True:
            action = agent.select_action(state)
            act = torch.zeros(1, 5, device=cfg.device)

            # Normalize the action vector
            action_np = action.cpu().numpy()
            norm = np.linalg.norm(action_np)
            action_np /= norm

            act[:, 1] = torch.from_numpy(action_np[0]).to(cfg.device)
            act[:, 3] = torch.from_numpy(action_np[1]).to(cfg.device)

            # Take a step in the environment
            next_obs_n, reward_n, done_n, _ = env.step(act.cpu().squeeze(0).numpy())
            next_obs = np.concatenate(next_obs_n, axis=0)

            # Preprocess the next observation
            image = env.render("rgb_array")[0]
            obs_img = img2obs(image)
            stacked_frames = stack_frame(stacked_frames, obs_img, False, cfg)
            next_state = normalize(np.concatenate(stacked_frames, axis=0))

            # Update the agent's replay buffer
            agent.buffer.rewards.append(reward_n[1])
            agent.buffer.is_terminals.append(done_n)

            state = next_state
            ep_reward += reward_n[1]
            i_step += 1

            print(f"Episode: {i_ep+1}/{cfg.train_eps}, Step {i_step}, Action: {action} Reward: {ep_reward:.3f}", end="\r")

            # Check for episode termination conditions
            if True in done_n or (i_step >= cfg.train_steps):
                break

        # Update the agent's policy network periodically
        if i_ep % 10 == 0:
            agent.update()

        rewards.append(ep_reward)

        # Update moving average rewards
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        # Save the model and results every 10 episodes
        if i_ep % 10 == 0:
            print('\nSave model')
            agent.save(cfg.model_path, i_ep)
            np.savetxt(cfg.model_path + 'reward_{}.txt'.format(curr_time), rewards)
            np.savetxt(cfg.model_path + 'ma_reward_{}.txt'.format(curr_time), ma_rewards)

    print('Training complete!')
    return rewards, ma_rewards

def eval(cfg: PPOConfig, env, agent):
    """
    Evaluate the model.

    Args:
        cfg (PPOConfig): Configuration object.
        env: The environment.
        agent: The agent.

    Returns:
        List: Rewards.
        List: Moving average rewards.
        int: Number of steps taken.
        bool: Whether the episode is done.
    """
    print('Start evaluation!')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    
    rewards = []
    ma_rewards = []  # moving average reward

    for i_ep in range(cfg.eval_eps):
        stacked_frames = []
        observation, info = env.reset()
        image = env.render("rgb_array")[0]
        obs_img = img2obs(image)
        stacked_frames = stack_frame(stacked_frames, obs_img, True, cfg)
        state = normalize(np.concatenate(stacked_frames, axis=0))
        
        ep_reward = 0
        i_step = 0

        while True:
            action = agent.select_action(state)
            act = torch.zeros(1, 5, device=cfg.device)

            # Normalize the action vector
            action_np = action.cpu().numpy()
            norm = np.linalg.norm(action_np)
            action_np /= norm

            act[:, 1] = torch.from_numpy(action_np[:,0]).to(cfg.device)
            act[:, 3] = torch.from_numpy(action_np[:,1]).to(cfg.device)

            next_obs_n, reward_n, done_n, _ = env.step(act.cpu().squeeze(0).numpy())

            image = env.render("rgb_array")[0]
            obs_img = img2obs(image)

            stacked_frames = stack_frame(stacked_frames, obs_img, False, cfg)
            state = normalize(np.concatenate(stacked_frames, axis=0))
            
            ep_reward += reward_n[1]
            i_step += 1

            print(f"Episode: {i_ep+1}/{cfg.eval_eps}, Action: {action}, Step {i_step} Reward: {ep_reward:.3f}", end="\r")

            if done_n[1]:
                print("DONE")
                
            if True in done_n or (i_step >= cfg.eval_steps):
                break

        print(f"\nEpisode: {i_ep+1}/{cfg.eval_eps}, Reward: {ep_reward:.3f}")
        rewards.append(ep_reward)

        # Update moving average rewards
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

    np.savetxt(cfg.model_path + 'reward_eval.txt', rewards)
    print('Evaluation complete!')
    return rewards, ma_rewards, i_step, done_n[1]


def img2obs(image_array):
    """Process an image to extract key information: goal position, agent position,
    pursuer position, and positions of the three nearest obstacles.

    Args:
        image_array (np.ndarray): BGR image with three channels.

    Returns:
        np.ndarray: Array containing the goal position, agent position,
        pursuer position, and positions of the three nearest obstacles.
    """

    # Resize the image for better processing
    pooled_image = cv2.resize(image_array, (800, 800), 0, 0, cv2.INTER_MAX)
    _, binary_dst = cv2.threshold(pooled_image[:, :, 0], 70, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_dst)

    # Define color ranges for goal, agent, and pursuer
    goal_lower_bound = np.array([199, 148, 199])
    goal_upper_bound = np.array([209, 158, 209])
    agent_lower_bound = np.array([167, 231, 167])
    agent_upper_bound = np.array([177, 241, 177])
    adv_lower_bound = np.array([231, 165, 165])
    adv_upper_bound = np.array([241, 177, 177])

    # Extract positions of goal, agent, and pursuer
    goal_im = np.array(np.where(np.all((pooled_image >= goal_lower_bound) & (pooled_image <= goal_upper_bound), axis=2))).transpose()[:, :2]
    agent_im = np.array(np.where(np.all((pooled_image >= agent_lower_bound) & (pooled_image <= agent_upper_bound), axis=2))).transpose()[:, :2]
    adv_im = np.array(np.where(np.all((pooled_image >= adv_lower_bound) & (pooled_image <= adv_upper_bound), axis=2))).transpose()[:, :2]

    agent_pos = np.mean(agent_im,axis=0).astype(int)
    agent_pos = np.array((agent_pos[1],agent_pos[0]))

    goal_pos = np.mean(goal_im,axis=0).astype(int)
    goal_pos = np.array((goal_pos[1],goal_pos[0]))

    adv_pos = np.mean(adv_im,axis=0).astype(int)
    adv_pos = np.array((adv_pos[1],adv_pos[0]))

    obstacle_pos = []
    distance = []

    # Calculate distances and positions of obstacles
    for i in range(num_labels):
        if i != 0:
            obstacle_pos.append(centroids[i].astype(int))  # Reverse the order for (row, col) convention
            distance.append(np.linalg.norm(agent_pos - centroids[i]))

    # Sort obstacle positions based on distance
    sorted_indexes = np.argsort(distance)

    # Normalize and concatenate the positions
    return np.concatenate((
        (goal_pos - agent_pos) / 256,
        agent_pos / 256,
        (adv_pos - agent_pos) / 256,
        (obstacle_pos[sorted_indexes[0]] - agent_pos) / 256,
        (obstacle_pos[sorted_indexes[1]] - agent_pos) / 256,
        (obstacle_pos[sorted_indexes[2]] - agent_pos) / 256
    ))