import gym
from gym import spaces
from rna import RNA
import numpy as np
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
import torch
from sb3_contrib import RecurrentPPO

torch.set_num_threads(20)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            obs = env2.reset_test()
            lstm_states = None
            num_envs = 1
            episode_starts = np.ones((num_envs,), dtype=bool)
            episode_rewards = []
            sum_rewards = 0
            for i in range(50):
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                obs, reward, done, info = env2.step_test(action)
                episode_starts = done
                episode_rewards.append(info['rmsd'])
                sum_rewards += info['rmsd']
                if done:
                    rewards_total.append(sum_rewards)
                    rewards.append(info['rmsd'])
                    rewards_angle.append(reward)

                    plt.clf()

                    plt.plot(episode_rewards)
                    plt.title(plot_title)
                    plt.xlabel('Timesteps')
                    plt.ylabel('-RMSD')
                    plt.savefig('episode_rmsd' + '.png')

                    plt.clf()

                    plt.plot(rewards)
                    plt.title(plot_title)
                    plt.xlabel('Evaluation #')
                    plt.ylabel('-RMSD')
                    plt.savefig('final_rmsd' + '.png')

                    plt.clf()

                    break

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(100,), dtype=np.float32) # Action outputs of angle perturbations.
        self.observation_space = spaces.Box(low=-180, high=180, shape=(196,), dtype=np.float32) # State inputs of angle readings and sequence encoding.
        self.counter = 0
        self.counter_test = 0

    def step(self, action):
        self.counter += 1
        action2 = action
        for k in range(len(action)):
            if action[k] < 0.5 and action[k] > -0.5:
                action2[k] = action[k]
            else:
                action2[k] = 0.0
        observation2, reward, done, info = env.step(action2)
        observation = observation2.flatten()
        if done == False:
            done = False
        if done == True:
            done = True
            self.counter = 0
        if self.counter == 50:
            done = True
            self.counter = 0
        return observation, reward, done, info

    def step_test(self, action):
        self.counter_test += 1
        action2 = action
        for k in range(len(action)):
            if action[k] < 0.5 and action[k] > -0.5:
                action2[k] = action[k]
            else:
                action2[k] = 0.0
        observation2, reward, done, info = env.step_test(action2)
        observation = observation2.flatten()
        if done == False:
            done = False
        if done == True:
            done = True
            self.counter_test = 0
        if self.counter_test == 50:
            done = True
            self.counter_test = 0
        return observation, reward, done, info

    def reset(self):
        ob = env.reset()
        observation = ob.flatten()
        return observation

    def reset_test(self):
        ob = env.reset_test()
        observation = ob.flatten()
        return observation

    def render(self, mode='human'):
        pass

    def close (self):
        pass

plot_title = 'All Backbone Angles'

env = RNA()

rewards = []
rewards_angle = []
rewards_total = []

env2 = CustomEnv()

policy_kwargs = dict(net_arch=dict(pi=[2048, 2048, 1024, 1024, 512, 512, 256, 256], qf=[2048, 2048, 1024, 1024, 512, 512, 256, 256])) # Model architecture.

model = RecurrentPPO('MlpLstmPolicy', env2, verbose=0, device="cuda:0", policy_kwargs = policy_kwargs)

callback = SaveOnBestTrainingRewardCallback(check_freq = 100) # Frequency of evaluations.
model.learn(total_timesteps = 500, callback=callback) # Training length.

model.save("policy")

obs = env2.reset_test()

lstm_states = None
num_envs = 1

episode_starts = np.ones((num_envs,), dtype=bool)

print('Testing final policy..')

for i in range(50):
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, reward, done, info = env2.step_test(action)
    episode_starts = done
    if done:
        print('Final RMSD:')
        print(info['rmsd'])
        break

