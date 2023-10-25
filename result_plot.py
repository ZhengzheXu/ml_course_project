import numpy as np
import matplotlib.pyplot as plt

# ma_reward = np.loadtxt("./saved_models/ma_reward_20230517-150629.txt")
# reward = np.loadtxt("./saved_models/reward_20230517-150629.txt")
#ma_reward = np.loadtxt("./saved_models/SpaceInvaders/ma_reward_20230519-101302.txt")
#reward = np.loadtxt("./saved_models/SpaceInvaders/reward_20230519-101302.txt")

# reward = np.loadtxt("ppo/saved_models/SpaceInvaders/save_PPO_model/img2obs_train/ma_reward_20230607-103436.txt")
# reward = np.loadtxt("/home/flipper/ml_project/ml_course_project/saved_models/My_Tag_0014/ma_reward_20231025-001629.txt")

# reward = np.loadtxt("/home/flipper/ml_project/ml_course_project/saved_models/My_Tag_Random_Img/reward_20231023-195922.txt")


# /home/flipper/ml_project/ml_course_project/saved_models/Simple_Tag_Random_Img/ma_reward_20231020-105613.txt
# reward = np.loadtxt("/home/flipper/ml_project/ml_course_project/saved_models/Simple_Tag_Random_Img/ma_reward_20231020-105613.txt")

# /home/flipper/ml_project/ml_course_project/saved_models/My_Tag_0014/ma_reward_20231025-004055.txt
reward = np.loadtxt("/home/flipper/ml_project/ml_course_project/saved_models/My_Tag_0014/reward_20231025-004055.txt")

max_reward_index = np.argsort(reward)[-100:]
print("max_reward_index = ", max_reward_index)
print("max_reward = ", reward[max_reward_index])


#plt.plot(ma_reward, label="ma_reward")
plt.plot(reward, label = "reward")
plt.legend()
plt.grid()
plt.show()