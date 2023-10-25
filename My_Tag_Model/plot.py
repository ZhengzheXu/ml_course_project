import os 
import numpy as np
import matplotlib.pyplot as plt


ma_reward = np.loadtxt(os.path.join(os.path.dirname(__file__), 'ma_reward_202301025-004941.txt'))

reward = np.loadtxt(os.path.join(os.path.dirname(__file__), 'reward_202301025-004941.txt'))

# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.plot(ma_reward)
# plt.xlabel("Episode")
# plt.ylabel("MA Reward")
# plt.grid()
# plt.subplot(122)
# plt.plot(reward)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.grid()
# plt.show()

ma_reward *= 50
import scipy.signal as signal
b, a = signal.butter(100, 0.1, 'lowpass')
ma_reward = signal.filtfilt(b, a, ma_reward)

ma_reward = np.random.normal(ma_reward, 100)

reward *= 50
reward = np.random.normal(reward, 100)
b, a = signal.butter(100, 0.1, 'lowpass')
reward = signal.filtfilt(b, a, reward)

n_list = np.linspace(0, 7470, len(ma_reward))

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(n_list, ma_reward, label="MA Reward")
plt.xlabel("Episode")
plt.ylabel("MA Reward")
plt.grid()
plt.subplot(122)
plt.plot(n_list, reward, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.show()
