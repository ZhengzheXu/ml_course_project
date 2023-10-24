import numpy as np

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import matplotlib.pyplot as plt 
import time

import casadi as ca

def mpc(env):
    # 获取agent的位置
    agent_pos = np.array([env.world.agents[0].state.p_pos]).reshape(1, 2)
    # 获取hunter的位置
    hunter_pos = np.array([env.world.agents[1].state.p_pos]).reshape(1, 2)
    # 获取agent的速度
    agent_vel = np.array([env.world.agents[0].state.p_vel]).reshape(1, 2)
    # 获取hunter的速度
    hunter_vel = np.array([env.world.agents[1].state.p_vel]).reshape(1, 2)
    # print(f"agent_pos: {agent_pos}, hunter_pos: {hunter_pos}, agent_vel: {agent_vel}, hunter_vel: {hunter_vel}")
    check_point = np.array([env.world.check[0].state.p_pos]).reshape(1, 2)
    # 获取landmark的位置
    lm1_pos = np.array([env.world.landmarks[0].state.p_pos]).reshape(1, 2)
    lm2_pos = np.array([env.world.landmarks[1].state.p_pos]).reshape(1, 2)
    lm3_pos = np.array([env.world.landmarks[2].state.p_pos]).reshape(1, 2)
    # 获取当前最近的landmark
    dist2lm = np.array([np.linalg.norm(agent_pos - lm1_pos), np.linalg.norm(agent_pos - lm2_pos), np.linalg.norm(agent_pos - lm3_pos)])
    lm_near_pos = np.array([lm1_pos, lm2_pos, lm3_pos])[np.argmin(dist2lm)]
    # print(f"lm_near_dist: {np.min(dist2lm)}")

    line_dis = 0.68
    ag2check = np.linalg.norm(agent_pos - check_point)

    N = 4
    T = 0.1

    hunter_pred = np.zeros((N, 2))
    for i in range(N):
        this_pos = hunter_pos + i*T*hunter_vel
        hunter_pred[i, :] = np.array([this_pos[0, 0], this_pos[0, 1]])
    # print(f"hunter_pred: {hunter_pred}")

    # 状态变量
    opti = ca.Opti()
    # 速度
    v = opti.variable(N, 2)
    # 位置
    p = opti.variable(N, 2)
    # 控制量
    u = opti.variable(N, 2) # 线加速度和方向
    # 速度约束
    # print(f"agent_vel: {agent_vel}")
    opti.subject_to(v[0, :] == agent_vel)
    # 位置约束
    opti.subject_to(p[0, :] == agent_pos)

    opti.subject_to(opti.bounded(0, u[:, 0], 1))
    # opti.subject_to(opti.bounded(0, u[:, 1], 2*np.pi))

    dis_safe = 0.03
    dis_safe_hunter = 0.02
    
    NN = 2
    for i in range(1, NN):
        # 约束条件:到landmark的距离大于安全距离
        opti.subject_to(ca.mtimes([(p[i, :] - lm_near_pos), (p[i, :] - lm_near_pos).T]) >= dis_safe**2)
        # 约束条件:到hunter的距离大于安全距离
        # opti.subject_to(ca.mtimes([(p[i, :] - hunter_pos), (p[i, :] - hunter_pos).T]) >= dis_safe_hunter**2)
        # print(f"p[i, :]: {p[i, :]}")
        # print(f"hunter_pred[i, :]: {hunter_pred[i, :]}")
        # print("shape of p[i, :]: ", p[i, :].shape)
        # print("shape of hunter_pred[i, :]: ", hunter_pred[i, :].shape)
        # print("shape of hunter_pred: ", hunter_pred.shape)
        opti.subject_to(ca.mtimes([(p[i, :] - hunter_pred[i, :].reshape(1, 2)), (p[i, :] - hunter_pred[i, :].reshape(1, 2)).T]) >= dis_safe_hunter**2)
    # 设置约束
    for i in range(N-1):
        opti.subject_to(p[i+1, :] == p[i, :] + T*v[i, :])
        opti.subject_to(v[i+1, 0] == v[i, 0] + T*u[i, 0]*ca.cos(u[i, 1]))
        opti.subject_to(v[i+1, 1] == v[i, 1] + T*u[i, 0]*ca.sin(u[i, 1]))
    
    # 代价为距离hunter的距离
    cost = 0
    Q_h = np.array([[1, 0], [0, 1]])
    Q_c = np.array([[1, 0], [0, 1]])
    R_u = np.array([[0.01, 0], [0, 1]])
    Q_o = np.array([[1, 0], [0, 1]])

    for i in range(N):
        # cost -= ca.mtimes([(p[i, :] - hunter_pos), Q_h, (p[i, :] - hunter_pos).T])
        # 到达check point的代价，越近越好
        if ag2check < line_dis:
            # print("到达check point")
            cost += ca.mtimes([(p[i, :] - check_point), Q_c, (p[i, :] - check_point).T]) * 5
        else:
            cost += ca.mtimes([(p[i, :] - check_point), Q_c, (p[i, :] - check_point).T]) * 0.6
        # cost += ca.mtimes([u[i, :], R_u, u[i, :].T]) * 0.02
        cost -= ca.mtimes([(p[i, :] - lm_near_pos), Q_o, (p[i, :] - lm_near_pos).T])
    
    opti.minimize(cost)
    settings = {'ipopt.print_level': 0, 'print_time': 0}
    opti.solver('ipopt', settings)
    # opti.solver('ipopt')
    # 设置代价函数

    # 求解
    sol = opti.solve()
    # print(f"sol: {sol.value(p)}")
    # print(f"sol: {sol.value(v)}")
    # print(f"sol: {sol.value(u)}")
    # print(f"sol: {sol.value(cost)}")
    return sol.value(u)
    
def get_state(env):
    agent_pos = np.array([env.world.agents[0].state.p_pos]).reshape(1, 2)
    agent_vel = np.array([env.world.agents[0].state.p_vel]).reshape(1, 2)
    print(f"agent_pos: {agent_pos}")
    print(f"agent_vel: {agent_vel}")
    return agent_pos, agent_vel

if __name__ == '__main__':
    # parse arguments
    scenario = scenarios.load("my_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.is_done, shared_viewer = True)

    # env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]))
    env.reset()
    image = env.render("rgb_array")  # read image
    step = 0
    total_step = 0
    max_step = 400
    # file = open('./course/label.txt', 'w')
    pos_x = []
    pos_y = []
    vel_x = []
    vel_y = []
    u_x = []
    u_y = []

    count = 0
    # while True:
    while count < 100:
        # [noop, move right, move left, move up, move down]
        # act_n = np.array([0,1,0,-5,0])
        input = mpc(env)
        this_input = input[0]
        # print(f"this_input: {this_input}")
        u_x = this_input[0] * np.cos(this_input[1])
        u_y = this_input[0] * np.sin(this_input[1])
        act_n = np.array([0, u_x, 0, u_y, 0])

        print("(u_x, u_y): ", np.around([u_x, u_y], decimals=3))
        # print(" u = ", np.linalg.norm([u_x, u_y]))
        # pos, vel = get_state(env)
        # pos_x.append(pos[0][0])
        # pos_y.append(pos[0][1])
        # vel_x.append(vel[0][0])
        # vel_y.append(vel[0][1])
        # u_x.append(act_n[1])
        # u_y.append(act_n[3])

        # 保留三位小数
        # print("input: ", np.around(input[0], decimals=3))
        next_obs_n, reward_n, done_n, _ = env.step(act_n, 1)
        # print(f"vel: {np.linalg.norm(next_obs_n[0][-4:-2])}, vel2: {np.linalg.norm(next_obs_n[0][-2:])}")
        image = env.render("rgb_array")[0]  # read image
        # print(f"shape: {np.shape(image)}")
        step += 1
        
        # try:
        #     input = mpc(agent, env)
        # except:
        #     pass
        if step % 50:
            # print(image)
            # plt.imshow(image)
            # plt.show()
            time.sleep(0.0167) # 60 fps
            # plt.close()

        if True in done_n or step > max_step:
            break
        # count += 1
    
    # print(f"pos_x: {pos_x}")
    # print(f"pos_y: {pos_y}")
    # print(f"vel_x: {vel_x}")
    # print(f"vel_y: {vel_y}")
    # dt = 0.1
    # time_list = np.arange(0, dt*len(pos_x), dt)
    # plt.figure()
    # # plt.plot(time_list, pos_x, label='pos_x')
    # # plt.plot(time_list, pos_y, label='pos_y')
    # plt.plot(time_list, vel_x, label='vel_x')
    # plt.plot(time_list, vel_y, label='vel_y')
    # # pred_vx = np.gradient(pos_x, dt)
    # # pred_vy = np.gradient(pos_y, dt)
    # # plt.plot(time_list, pred_vx, label='pred_vx')
    # # plt.plot(time_list, pred_vy, label='pred_vy')
    # pred_ax = np.gradient(vel_x, dt)
    # pred_ay = np.gradient(vel_y, dt)
    # plt.plot(time_list, pred_ax, label='pred_ax')
    # plt.plot(time_list, pred_ay, label='pred_ay')
    # plt.legend()
    # plt.show()