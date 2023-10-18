import sys,os
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path

import torch
import datetime
import time
import math

from agents.PPO.agent import PPO
import numpy as np

import gymnasium as gym

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import cv2

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time

class PPOConfig:
    def __init__(self) -> None:
        self.algo = 'PPO'
        # self.env = "CartPole-v1"
        self.env = "MULTIAGENT-ENVS"
        self.seed = 15
        self.ShowImage = False      # render image
        # self.result_path = curr_path+"/outputs/" +self.env+'/'+curr_time+'/results/'  # path to save results
        self.load_model = False     # load model
        self.train = False          # train model
        self.model_path = 'saved_models/Simple_Tag_Random_Img/'  # path to save models
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.capacity = int(2e5)    # replay buffer size
        self.batch_size  = 512      # minibatch size
        self.gamma = 0.99       # discount factor
        self.lr = 1e-4          # learning rate
        self.eps_clip = 0.2
        self.K_epochs = 80
        self.update_every = 1   # Learn every UPDATE_EVERY time steps.
        self.train_eps = 10000
        self.train_steps = 200
        self.eval_eps = 1
        self.eval_steps = 2000
        self.eps_start = 1.0
        self.eps_decay = 0.995
        self.eps_end = 0.01
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frames = 10

def env_agent_config(cfg:PPOConfig):
    """
    Create env and agent
    ------------------------
    Input: configuration
    Output: env, agent
    """
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.is_done, shared_viewer = True)

    # [noop, move right, move left, move up, move down] 这里只选择两个动作，向右和向上，[0,1,0,1,0]
    action_dim = 2
    # 这个地方需要修改，首先把图片里的信息提取出来，然后再把这些信息输入到网络里，包括小球和目标点的位置信息，这些信息就是state
    state_len = 1

    state_dim = 12
    # for i in range(state_len):
    state_dim = state_dim * state_len
    
    print(f"action dim: {action_dim}, state dim: {state_dim}")
    N = cfg.frames 
    agent = PPO(state_dim * N,action_dim,cfg)
    # print(agent.qnetwork_local)
    return env,agent

def normalize(state, observation_space = None):
    norm = np.linalg.norm(state)
    normalized = state / norm
    return normalized

def stack_frame(stacked_frames, frame, is_new, cfg:PPOConfig):
    if is_new:
        for i in range(cfg.frames):
            stacked_frames.append(frame)
    else:
        for i in range(cfg.frames - 1):
            stacked_frames[i] = stacked_frames[i+1]
        
        stacked_frames[-1] = frame
    # print(f"diff: {(stacked_frames[-1] - stacked_frames[0]) / 255}")
    return stacked_frames

def train(cfg:PPOConfig,env,agent):
    """
    Train model, and save agent and result during training.
    --------------------------------
    Input: configuration, env, agent
    Output: reward and ma_reward
    """
    print('Start to train !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward

    eps = cfg.eps_start
    for i_ep in range(cfg.train_eps):
        
        observation, info = env.reset()

        image = env.render("rgb_array")[0]  
        obs_img = img2obs(image)
        stacked_frames = [obs_img]*cfg.frames
        stacked_frames = stack_frame(stacked_frames, obs_img, False, cfg)
        state = normalize(np.concatenate(stacked_frames, axis=0))
        # state = normalize(np.concatenate((stacked_frames[0], stacked_frames[-1]), axis=0))
        ep_reward = 0
        i_step = 0
        
        while True:
            print(i_step)
            action = agent.select_action(state)
            act = torch.zeros(1,5,device=cfg.device)

            action_np = action.cpu().numpy()
            norm = math.sqrt(action_np[:,0]**2+action_np[:,1]**2)
            action_np[:,0] = action_np[:,0]/norm
            action_np[:,1] = action_np[:,1]/norm

            act[:,1] = torch.from_numpy(action_np[:,0]).to(cfg.device)
            act[:,3] = torch.from_numpy(action_np[:,1]).to(cfg.device)
            # import ipdb;ipdb.set_trace()
            next_obs_n, reward_n, done_n, _ = env.step(act.cpu().squeeze(0).numpy())
            next_obs = np.concatenate(next_obs_n, axis=0)

            image = env.render("rgb_array")[0]  
            obs_img = img2obs(image)
            """
            处理图像数据，提取出图片中的状态信息
            """


            stacked_frames = stack_frame(stacked_frames, obs_img, False, cfg)
            next_state = normalize(np.concatenate(stacked_frames, axis=0))
            agent.buffer.rewards.append(reward_n[1])
            agent.buffer.is_terminals.append(done_n)
            # import ipdb;ipdb.set_trace()

            state = next_state
            ep_reward += reward_n[1]
            i_step += 1

            # print(f"episode: {i_ep}, step: {i_step}", end="\r")
            # print(f"action: {action}, real_action: {action_in}")
            # print(f"state: {state}", end="\n")
            print(f"Episode:{i_ep+1}/{cfg.train_eps}, step {i_step}, action: {action} Reward:{ep_reward:.3f}", end="\r")
            if True in done_n or (i_step >= cfg.train_steps):
                # if done_n[1]:
                #     time.sleep(1)
                #     print("SUCCESS")
                break
        if i_ep % 8 == 0:
            agent.update()

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 

   
        if i_ep % 8 == 0:
            print('\nsave model')
            agent.save(cfg.model_path,i_ep)
            np.savetxt(cfg.model_path+'reward_{}.txt'.format(curr_time),rewards)
            np.savetxt(cfg.model_path+'ma_reward_{}.txt'.format(curr_time),ma_rewards)

    print('Complete training！')
    return rewards, ma_rewards

def eval(cfg:PPOConfig,env,agent):
    """
    Evaluate Model
    """
    print('Start to eval !')
    print(f'Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}')
    rewards  = []
    ma_rewards = [] # moveing average reward

    for i_ep in range(cfg.eval_eps):
        stacked_frames = []
        observation, info = env.reset()
        image = env.render("rgb_array")[0]  
        obs_img = img2obs(image)
        stacked_frames = stack_frame(stacked_frames, obs_img, True, cfg)
        state = normalize(np.concatenate(stacked_frames, axis=0))
        # state = normalize(np.concatenate((stacked_frames[0], stacked_frames[-1]), axis=0))
        ep_reward = 0
        i_step = 0
        while True:
        # for i_step in range(cfg.eval_steps):
            action = agent.select_action(state)
            # import ipdb;ipdb.set_trace()
            act = torch.zeros(1,5,device=cfg.device)

            action_np = action.cpu().numpy()
            norm = math.sqrt(action_np[:,0]**2+action_np[:,1]**2)
            action_np[:,0] = action_np[:,0]/norm
            action_np[:,1] = action_np[:,1]/norm

            act[:,1] = torch.from_numpy(action_np[:,0]).to(cfg.device)
            act[:,3] = torch.from_numpy(action_np[:,1]).to(cfg.device)
            next_obs_n, reward_n, done_n, _ = env.step(act.cpu().squeeze(0).numpy())

            image = env.render("rgb_array")[0]  
            obs_img = img2obs(image)

            stacked_frames = stack_frame(stacked_frames, obs_img, False, cfg)
            state = normalize(np.concatenate(stacked_frames, axis=0))
            # state = normalize(np.concatenate((stacked_frames[0], stacked_frames[-1]), axis=0))
            ep_reward += reward_n[1]
            i_step += 1

            print(f"Episode:{i_ep+1}/{cfg.eval_eps}, action: {action}, step {i_step} Reward:{ep_reward:.3f}", end="\r")
            # print(f"\rreward: {reward}", end="")
            # print(f"action: {action}, real_action: {action_in}")
            if done_n[1]:
                print("DONE")
            if True in done_n or (i_step >= cfg.eval_steps):
                break

        print(f"\nEpisode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward) 
    np.savetxt(cfg.model_path+'reward_eval.txt',rewards)
    print('Complete evaling！')
    return rewards, ma_rewards,i_step,done_n[1]

def img2obs(image_array):
    """处理图片获得关键信息,目标位置，智能体位置，追击者位置，障碍物位置(最近的3个障碍物位置)

    Args:
        image_array (nparray): 三通道的bgr图片

    Returns:
        obs (): 目标位置，智能体位置，追击者位置，障碍物位置(最近的3个障碍物位置)
    """
    obstacle_num_in_obs = 3

    pooled_image = cv2.resize(image_array, (800,800), 0, 0, cv2.INTER_MAX)
    _, binary_dst = cv2.threshold(pooled_image[:,:,0], 70, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_dst)

    # 显示池化后的图像
    #cv2.imshow('Pooled Image', pooled_image)
    # cv2.imshow('image_array', image_array)
    # cv2.imshow('THRESH_BINARY', binary_dst)
    

    # 定义颜色范围
    #204,153,204
    goal_lower_bound = np.array([199, 148, 199])
    goal_upper_bound = np.array([209, 158, 209])
    #172,236,172
    agent_lower_bound = np.array([167, 231, 167])
    agent_upper_bound = np.array([177, 241, 177])
    #172,172,236
    adv_lower_bound = np.array([231,165, 165])
    adv_upper_bound = np.array([241,177, 177])

    goal_im = np.array(np.where(np.all((pooled_image>=goal_lower_bound) & (pooled_image<=goal_upper_bound),axis=2))).transpose()[:,:2]
    agent_im = np.array(np.where(np.all((pooled_image>=agent_lower_bound) & (pooled_image<=agent_upper_bound),axis=2))).transpose()[:,:2]
    adv_im = np.array(np.where(np.all((pooled_image>=adv_lower_bound) & (pooled_image<=adv_upper_bound),axis=2))).transpose()[:,:2]

    # cv2.imshow('goal_im', pooled_image[:,:,0] * np.all((pooled_image>=goal_lower_bound) & (pooled_image<=goal_upper_bound),axis=2))
    # cv2.imshow('agent_im', pooled_image[:,:,0] * np.all((pooled_image>=agent_lower_bound) & (pooled_image<=agent_upper_bound),axis=2))
    # cv2.imshow('adv_im', pooled_image[:,:,0] * np.all((pooled_image>=adv_lower_bound) & (pooled_image<=adv_upper_bound),axis=2))
     #将一维灰度图像扩展到三维
    # labels= np.expand_dims(binary_dst,axis=2).repeat(3,axis=2).astype(np.uint8)
    # for st in stats[1:]:
    #     cv2.rectangle(labels, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 1)
    # cv2.imshow('labels', labels)

    # contours,hierarchy = cv2.findContours(pooled_image[:,:,0] * np.all((pooled_image>=goal_lower_bound) & (pooled_image<=goal_upper_bound),axis=2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    # for obj in contours:
    #     area = cv2.contourArea(obj)  #计算轮廓内区域的面积
    #     #cv2.drawContours(image_array, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
    #     perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
    #     approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
    #     CornerNum = len(approx)   #轮廓角点的数量
    #     x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度

    #     #轮廓对象分类
    #     if CornerNum ==3: objType ="triangle"
    #     elif CornerNum == 4:
    #         if w==h: objType= "Square"
    #         else:objType="Rectangle"
    #         cv2.rectangle(pooled_image,(x,y),(x+w,y+h),(0,0,255),5)  #绘制边界框
    #         #cv2.putText(image_array,objType,(x+(w//2),y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),1)  #绘制文字
    #     elif CornerNum>4: objType= "Circle"
    #     else:objType="N"

        

    # cv2.imshow('rectangle', pooled_image)


    #cv2.waitKey(0)

    agent_pos = np.mean(agent_im,axis=0).astype(int)
    agent_pos = np.array((agent_pos[1],agent_pos[0]))

    goal_pos = np.mean(goal_im,axis=0).astype(int)
    goal_pos = np.array((goal_pos[1],goal_pos[0]))
    adv_pos = np.mean(adv_im,axis=0).astype(int)
    adv_pos = np.array((adv_pos[1],adv_pos[0]))

    obstacle_pos = []
    distance = []

    for i in range(num_labels):
        if i!=0:
            obstacle_pos.append(centroids[i].astype(int))
            distance.append(np.linalg.norm(agent_pos-centroids[i]))
    
    #获取distance按大小排序后的index
    sorted_indexes = np.argsort(distance)
    #return np.concatenate((goal_pos,agent_pos,adv_pos,obstacle_pos[sorted_indexes[0]],obstacle_pos[sorted_indexes[1]],obstacle_pos[sorted_indexes[2]]))/256
    return np.concatenate((goal_pos-agent_pos,agent_pos,adv_pos-agent_pos,obstacle_pos[sorted_indexes[0]]-agent_pos,obstacle_pos[sorted_indexes[1]]-agent_pos,obstacle_pos[sorted_indexes[2]]-agent_pos))/256
  

if __name__ == "__main__":
    cfg=PPOConfig()
    
    env,agent = env_agent_config(cfg)

    # print("env.action_space",env.action_space)
    # print("env.action_space.n",env.action_space.n)
    # print("env.observation_space",env.observation_space)
    # print("env.observation_space.shape[0]",env.observation_space.shape)

    if cfg.train:
        # train
        if cfg.load_model:
            print(">>>>>>>>>>load model<<<<<<<<<<<<<<<")
            agent.load(path=cfg.model_path,i_ep=1016)
        rewards, ma_rewards = train(cfg, env, agent)
    else:
        # eval
        print(">>>>>>>>>>load model<<<<<<<<<<<<<<<")
        epoch = 0
        ss_epoch = 0
        i_step_all = 0
        for i in range(3280,3320,8):
            print(i)
            epoch +=1
            agent.load(path=cfg.model_path,i_ep=i)
            rewards, ma_rewards,i_step,done = eval(cfg, env, agent)
            if done:
                i_step_all += i_step
                ss_epoch +=1
        print("成功率：",ss_epoch/epoch)
        print("平均步数",i_step_all/ss_epoch)