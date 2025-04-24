import numpy as np
from scipy.special import softmax
import cv2
import collections

import matplotlib
import matplotlib.pyplot as plt

from pyvirtualdisplay import Display
from tqdm.auto import tqdm

import os



import torch
import torch.nn as nn
import torch.optim as optim

import gym
import gym.spaces


# 使用するデバイス（GPU or CPU）の決定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use device:", device)



env = gym.make('CartPole-v1')

obs = env.reset()
print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('initial observation', obs)

print(env.spec)
print(env.reward_range)
print(env.metadata)

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, dim=64):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim * dim),
            nn.LeakyReLU(),
            nn.Linear(dim * dim, n_actions)
        )

    def forward(self, x):
        return self.fc(x)


def get_action_qn(next_state, net, episode, device="cpu"):
    # epsilon = 0.5 * (1 / (episode + 1))
    epsilon = 0.8
    if np.random.uniform(0, 1) <= epsilon:
        state_a = np.array([next_state], copy=False)
        state_v = torch.tensor(state_a).float().to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        next_action = int(act_v.item())
    else:
        next_action = np.random.choice([0, 1])

    return next_action


def calc_loss(batch, net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states = torch.tensor(states, device=device)
    next_states = torch.tensor(next_states, device=device)
    actions = torch.tensor(actions, device=device, dtype=torch.int64)
    rewards = torch.tensor(rewards, device=device)
    dones = torch.tensor(dones, device=device, dtype=torch.int64)

    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_state_values = net(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = rewards + GAMMA * next_state_values
    return nn.MSELoss()(state_action_values, expected_state_action_values)


Experience = collections.namedtuple(
    'Experience', 
    field_names=['state', 'action', 'reward', 'done', 'new_state'])

class Buffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):

        """
        (('state1', 'action1', 'reward1', 'done1', 'new_state1'),
         ('state2', 'action2', 'reward2', 'done2', 'new_state2'), ...)
        -->
        [('state1', 'state2', ...), 
         ('action1', 'action2', ...), 
          ..., 
         ('new_state1', 'new_state2', ...)]
        """
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in range(batch_size)])

        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)


max_number_of_steps = 500  #1試行のstep数
num_episodes = 50000  #総試行回数

LEARNING_RATE = 0.05 #学習率
GAMMA = 0.99 #割引率

batch_size = 4 #バッチサイズ
# device = 'cuda:0'
train_num = 0

net = QNetwork(env.observation_space.shape[0],
               env.action_space.n).to(device)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
buffer = Buffer(batch_size * 10)



with tqdm(range(num_episodes)) as pbar_episodes:

    reward_list = []  # 各試行の報酬を格納
    loss_list = []

    for episode in pbar_episodes:  #試行数分繰り返す

        # 環境の初期化
        observation = env.reset()
        state = observation
        episode_reward = 0
        episode_loss_list = []

        for t in range(max_number_of_steps):  #1試行のループ

            action = get_action_qn(observation, net, episode, device)
            
            # 行動a_tの実行により、s_{t+1}, r_{t}などを計算する
            observation, reward, done, info = env.step(action)
            next_state = observation

            episode_reward += reward  # 報酬を追加
            
            exp = Experience(state, action, reward, done, next_state)
            buffer.append(exp)
            state = next_state

            if train_num >= batch_size:
                optimizer.zero_grad()
                batch = buffer.sample(batch_size)
                loss_t = calc_loss(batch, net, device=device)
                loss_t.backward()
                optimizer.step()

                train_num = 0
                episode_loss_list.append(loss_t.item())
                
            train_num += 1

            if done:
                break


        # 終了時の処理
        reward_list.append(episode_reward)  # 報酬を記録
        if len(episode_loss_list) > 0:
            loss_list.append(np.mean(episode_loss_list))


        pbar_episodes.set_postfix_str(
            'reward={:.05f} loss={:.05f}'.format(np.mean(reward_list), np.mean(loss_list)))
