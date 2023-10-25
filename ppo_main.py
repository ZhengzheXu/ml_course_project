#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ppo_function import PPOConfig, env_agent_config, train, eval

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