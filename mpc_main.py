import numpy as np

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import matplotlib.pyplot as plt 
import time

import casadi as ca

def extract_positions_velocities(env):
    agent_pos = np.array([env.world.agents[0].state.p_pos])
    hunter_pos = np.array([env.world.agents[1].state.p_pos])
    agent_vel = np.array([env.world.agents[0].state.p_vel])
    hunter_vel = np.array([env.world.agents[1].state.p_vel])
    check_point = np.array([env.world.check[0].state.p_pos])
    lm_positions = np.array([lm.state.p_pos for lm in env.world.landmarks])

    return agent_pos, hunter_pos, agent_vel, hunter_vel, check_point, lm_positions

def find_nearest_landmark(agent_pos, lm_positions):
    dist_to_lm = np.linalg.norm(agent_pos - lm_positions, axis=1)
    nearest_lm_pos = lm_positions[np.argmin(dist_to_lm)]
    nearest_lm_pos = np.array([nearest_lm_pos])
    return nearest_lm_pos

def predict_hunter_positions(hunter_pos, hunter_vel, N, T):
    hunter_pred = np.zeros((N, 2))
    for i in range(N):
        this_pos = hunter_pos + i * T * hunter_vel
        hunter_pred[i, :] = this_pos[0, :]
    return hunter_pred

def setup_optimization_problem(agent_pos, agent_vel, N, T, check_point, line_dis, nearest_lm_pos, hunter_pred):
    opti = ca.Opti()

    v = opti.variable(N, 2)
    p = opti.variable(N, 2)
    u = opti.variable(N, 2)

    opti.subject_to(v[0, :] == agent_vel)
    opti.subject_to(p[0, :] == agent_pos)
    opti.subject_to(opti.bounded(0, u[:, 0], 1))

    dis_safe = 0.03
    dis_safe_hunter = 0.02

    for i in range(1, N):
        if np.linalg.norm(agent_pos - check_point) < line_dis:
            pass
        else:
            opti.subject_to(ca.mtimes([(p[i, :] - nearest_lm_pos), (p[i, :] - nearest_lm_pos).T]) >= dis_safe ** 2)
        opti.subject_to(
            ca.mtimes([(p[i, :] - hunter_pred[i, :].reshape(1, 2)), (p[i, :] - hunter_pred[i, :].reshape(1, 2)).T]) >=
            dis_safe_hunter ** 2)

    for i in range(N - 1):
        opti.subject_to(p[i + 1, :] == p[i, :] + T * v[i, :])
        opti.subject_to(v[i + 1, 0] == v[i, 0] + T * u[i, 0] * ca.cos(u[i, 1]))
        opti.subject_to(v[i + 1, 1] == v[i, 1] + T * u[i, 0] * ca.sin(u[i, 1]))

    var = (v, p, u)
    return opti, var

def define_cost_function(opti, var, N, agent_pos, check_point, line_dis, nearest_lm_pos):
    cost = 0
    v, p, u = var
    Q_h = np.array([[1, 0], [0, 1]])
    Q_c = np.array([[1, 0], [0, 1]])
    R_u = np.array([[0.01, 0], [0, 1]])
    Q_o = np.array([[1, 0], [0, 1]])

    for i in range(N):
        if np.linalg.norm(agent_pos - check_point) < line_dis:
            cost += ca.mtimes([(p[i, :] - check_point), Q_c, (p[i, :] - check_point).T]) * 5
        else:
            cost += ca.mtimes([(p[i, :] - check_point), Q_c, (p[i, :] - check_point).T]) * 0.6
        cost -= ca.mtimes([(p[i, :] - nearest_lm_pos), Q_o, (p[i, :] - nearest_lm_pos).T])

    return cost

def mpc(env):
    agent_pos, hunter_pos, agent_vel, hunter_vel, check_point, lm_positions = extract_positions_velocities(env)
    nearest_lm_pos = find_nearest_landmark(agent_pos, lm_positions)

    N = 4
    T = 0.1
    line_dis = 0.7

    hunter_pred = predict_hunter_positions(hunter_pos, hunter_vel, N, T)

    opti, var = setup_optimization_problem(agent_pos, agent_vel, N, T, check_point, line_dis, nearest_lm_pos, hunter_pred)
    v, p, u = var
    cost = define_cost_function(opti, var, N, agent_pos, check_point, line_dis, nearest_lm_pos)

    opti.minimize(cost)
    settings = {'ipopt.print_level': 0, 'print_time': 0}
    opti.solver('ipopt', settings)

    sol = opti.solve()
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

    dt = 0.1
    u_list = []
    x_list = []

    while True:
        input = mpc(env)
        this_input = input[0]

        u_x = this_input[0] * np.cos(this_input[1])
        u_y = this_input[0] * np.sin(this_input[1])
        act_n = np.array([0, u_x, 0, u_y, 0])

        print("(u_x, u_y): ", np.around([u_x, u_y], decimals=3))

        next_obs_n, reward_n, done_n, _ = env.step(act_n, 1)
        image = env.render("rgb_array")[0]  # read image
        step += 1

        u_list.append([u_x, u_y])
        x_list.append([env.world.agents[0].state.p_pos[0], env.world.agents[0].state.p_pos[1]])

        if step % 50:
            time.sleep(0.0167) # 60 fps

        if True in done_n or step > max_step:
            break
    
    time_list = np.arange(0, len(u_list) * dt, dt)
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    plt.plot(time_list, np.array(x_list)[:, 0], label="$x$")
    plt.plot(time_list, np.array(x_list)[:, 1], label="$y$")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid()
    plt.subplot(122)
    plt.plot(time_list, np.array(u_list)[:, 0], label="$u_x$")
    plt.plot(time_list, np.array(u_list)[:, 1], label="$u_y$")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.grid()
    plt.show()
    