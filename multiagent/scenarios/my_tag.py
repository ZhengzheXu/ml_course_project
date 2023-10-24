import numpy as np
from multiagent.core import World, Agent, Landmark, Border, Check
from multiagent.scenario import BaseScenario
import math


class Scenario(BaseScenario):
    def make_world(self):
        # global index
        # index = 0
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 1
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 3
        num_check = 1
        num_borders = 80  # (20 * 4) 4条边，每边20个border
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):  # agent的设置
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.025 if agent.adversary else 0.025
            agent.accel = 0.85 if agent.adversary else 1.0  # 加速度
            # agent.accel = 1
            # agent.accel = 20.0 if agent.adversary else 25.0
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.max_speed = 0.28 if agent.adversary else 0.25
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False
        world.check = [Check() for i in range(num_check)]
        for i, check in enumerate(world.check):
            check.name = 'checkpoint %d' % i
            check.collide = False
            check.movable = False
            check.size = 0.05
            check.boundary = False
            check.shape = [[-0.025, -0.025], [0.025, -0.025],
                           [0.025, 0.025], [-0.025, 0.025]]
        # 加入 borders
        world.borders = [Border() for i in range(num_borders)]
        for i, border in enumerate(world.borders):
            border.name = 'border %d' % i
            border.collide = True
            border.movable = False
            border.size = 0.15  # 边界大小
            border.boundary = True
            # 改变边界厚度border.shape
            border.shape = [[-0.05, -0.05], [0.05, -0.05],
                            [0.05, 0.05], [-0.05, 0.05]]

            #print(border.pos)
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, agent_pos=None):
        # random properties for agents
        for i, agent in enumerate(world.agents):  # agent颜色
            agent.color = np.array(
                [0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])  # landmark 颜色
        # # random properties for borders
        for i, border in enumerate(world.borders):
            border.color = np.array([0.8, 0.4, 0.4])  # 边界颜色
        # # set random initial
        for i, check in enumerate(world.check):
            check.color = np.array([0.8, 0.6, 0.8])
        for agent in world.agents:
            # agent初始位置状态 [x,y]=[0,0]是在视野中心
            # agent.state.p_pos = np.random.uniform(-0.28, -0.05, world.dim_p) if agent.adversary else np.random.uniform(0.05, 0.28, world.dim_p)
            if agent_pos is None:
                agent.state.p_pos = np.array(
                    [0.0, 0.0]) if agent.adversary else np.array([0.5, 0.5])
            else:
                agent.state.p_pos = agent_pos[0] if agent.adversary else agent_pos[1] 
            agent.state.p_vel = np.zeros(world.dim_p)  # agent初始速度
            agent.state.c = np.zeros(world.dim_c)  # agent初始交流状态
        pos = [[-0.35, 0.35], [0.35, 0.35], [0, -0.35]]
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = pos[i]  # landmark初始位置
                landmark.state.p_vel = np.zeros(world.dim_p)
        # world.check[0].state.p_pos = [np.random.uniform(-0.6,0.6),np.random.uniform(-0.6,0.6)]
        world.check[0].state.p_pos = [-0.5, -0.5]
        world.check[0].state.p_vel = np.zeros(world.dim_p)
        # # 增加部分 [x,y]=[0,0]是在视野中心
        # # 每条边20个border， 计算好大概位置，依次为每条边的border生成位置坐标
        pos = []
        x = -0.95
        y = -1.0
        # bottom
        for count in range(20):
            pos.append([x, y])
            x += 0.1

        x = 1.0
        y = -0.95
        # right
        for count in range(20):
            pos.append([x, y])
            y += 0.1

        x = 0.95
        y = 1.0
        # top
        for count in range(20):
            pos.append([x, y])
            x -= 0.1

        x = -1.0
        y = 0.95
        # left
        for count in range(20):
            pos.append([x, y])
            y -= 0.1

        for i, border in enumerate(world.borders):
            border.state.p_pos = np.asarray(pos[i])  # 将设好的坐标传到border的位置坐标
            border.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def is_done(self, agent, world):
        # agent
        if agent.adversary:
            good_agent = self.good_agents(world)[0]
            if self.is_collision(good_agent, agent):
                return True

        #landmark
        if not agent.adversary:
            for i, landmark in enumerate(world.landmarks):
                delta_dis = agent.state.p_pos - landmark.state.p_pos 
                dist = np.sqrt(np.sum(np.square(delta_dis)))
                # dist_min = agent.size + landmark.size
                dist_min = 0.075
                if dist <= dist_min :
                    return True
            #check success
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.check[0].state.p_pos)))
            if dist < agent.size + world.check[0].size:
                return True

    def huber_loss(self, x, delta=1.0):
        if abs(x) < delta:
            return 0.5 * x ** 2
        else:
            return delta * (abs(x) - 0.5 * delta)
        
    def lin_squ_loss(self, x, delta=1.0):
        if abs(x) < delta:
            return x
        else:
            # 平方损失
            m = 0.5/delta
            return m * (x ** 2) - 0.5 * delta

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        adversaries = self.adversaries(world)
        # 碰撞惩罚
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 3000
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                if self.is_collision(landmark, agent):
                    rew -= 3000
        for i, border in enumerate(world.borders):
            if self.is_collision(border, agent):
                rew -= 3000

        dist = np.sqrt(
            np.sum(np.square(agent.state.p_pos - world.check[0].state.p_pos)))
        # 距离check点越远，惩罚越大
        # check_rew = - 2 * self.lin_squ_loss(dist, delta=0.4)
        # rew += check_rew

        agent_to_landmark = []
        for i, landmark in enumerate(world.landmarks):
            agent_to_landmark.append(np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))))
        # 距离landmark越远,奖励越大
        # 选出最小的N个距离
        N = 1
        agent_to_landmark.sort()
        avg_dis = sum(agent_to_landmark[:N])/N
        # print("agent_to_landmark:", agent_to_landmark)
        # ag2land_rew = 100 * self.huber_loss(avg_dis, delta=0.1)
        # rew += ag2land_rew

        # print("check_rew:", check_rew)
        # print("ag2land_rew:", ag2land_rew)
        if dist < agent.size + world.check[0].size:
            # 完成任务的奖励
            # print("success")
            rew += 5000
        
        # 速度方向惩罚
        v_ad = adversaries[0].state.p_vel
        v_ag = agent.state.p_vel
        ad2ag = agent.state.p_pos - adversaries[0].state.p_pos
        
        ag_proj = np.dot(ad2ag, v_ag) / np.linalg.norm(ad2ag)
        ad_proj = np.dot(ad2ag, v_ad) / np.linalg.norm(ad2ag)
        
        # 奖励ag_proj
        ag_reward = ag_proj / np.linalg.norm(v_ag)
        ag_reward = 10 * np.sign(ag_reward)
        # 惩罚ad_proj
        ad_reward = - ad_proj / np.linalg.norm(v_ad)
        ad_reward = 10 * np.sign(ad_reward)

        # rew += ag_reward
        # rew += ad_reward
        # print("ag_reward:", ag_reward)
        # print("ad_reward:", ad_reward)

        # 计算agent到check点的向量
        agent_to_check = world.check[0].state.p_pos - agent.state.p_pos
        # 计算距离
        dist = np.sqrt(np.sum(np.square(agent_to_check)))
        safe_dist = 0.8
        # if dist < safe_dist:
        #     print("In safe region")
        #     # 奖励速度方向
        #     v_ag = agent.state.p_vel
        #     ag_proj = np.dot(agent_to_check, v_ag) / np.linalg.norm(agent_to_check)
        #     ag_reward = ag_proj / np.linalg.norm(v_ag)
        #     rew += 0.5 * ag_reward
        #     print("v_diff_reward:", 0.1 * ag_reward)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # reward can optionally be shaped (decreased reward for increased distance from agents)
        if shape:
            for adv in adversaries:
                # adversary reward = 0.1 * min(adv与agent之间的距离), 即越远越好
                this_rew = 2 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)) + 1e-6)for a in agents])
                # print("adv_rew:", this_rew)
                # rew += this_rew
                
                obs_dis = []
                for obs in world.landmarks:
                    if not obs.boundary: # 
                        obs_dis.append(np.sqrt(np.sum(np.square(obs.state.p_pos - adv.state.p_pos))))
                
                # 越近越好
                # obs_dis_min = min(obs_dis)
                # if obs_dis_min < 0.01:
                #     rew += 0.2
                # else:
                #     rew -= 0.0005 * obs_dis_min
                    
        # if agent.collide:
        #     for ag in agents:
        #         for adv in adversaries:
        #             if self.is_collision(ag, adv):
        #                 rew -= 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        check_pos = []
        check_pos.append(agent.state.p_pos - world.check[0].state.p_pos)
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary:
            #     other_vel.append(other.state.p_vel)
            other_vel.append(other.state.p_vel)  # 增加
            dists = np.sqrt(np.sum(np.square(agent.state.p_pos - other_pos)))
        # return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + dists)   # 知道自己的位置、速度和与其他agent的距离
        return np.concatenate([agent.state.p_pos] + other_pos + check_pos + entity_pos + [agent.state.p_vel] + other_vel)
