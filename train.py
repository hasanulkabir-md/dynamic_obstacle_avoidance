
"""
train.py
-----------------------------------------
Main training script for video-based dynamic obstacle avoidance
using Deep Reinforcement Learning (Double DQN + YOLO perception).

Author: Md. Hasanul Kabir
Nanjing Normal University (Project 211)
Email: hasanul.kabir09@gmail.com
"""

import os
import cv2
import gym
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime

# -------------------------------
# Parameters
# -------------------------------
EPISODES = 10000
MAX_STEPS = 500
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE = 100
SAVE_INTERVAL = 500

LOG_DIR = "experiments/logs/run_2025_01"
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Neural Network (Double DQN)
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


# -------------------------------
# Agent Class
# -------------------------------
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# -------------------------------
# Main Training Loop
# -------------------------------
def train(env_name="CartPole-v1"):  # Replace with GazeboEnv or custom ROS2 environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DoubleDQNAgent(state_dim, action_dim)

    reward_history = []
    loss_history = []

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        losses = []

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            if loss:
                losses.append(loss)
            state = next_state
            episode_reward += reward
            if done:
                break

        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-50:])

        # Epsilon decay
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        # Save model
        if episode % SAVE_INTERVAL == 0:
            torch.save(agent.policy_net.state_dict(), f"{LOG_DIR}/model_ep{episode}.pth")

        # Log training progress
        log_line = f"Episode {episode:5d} | Reward: {episode_reward:8.2f} | Avg: {avg_reward:8.2f} | Eps: {agent.epsilon:.3f}"
        print(log_line)
        with open(f"{LOG_DIR}/summary.txt", "a") as f:
            f.write(log_line + "\n")

    # Save reward history
    np.savetxt(f"{LOG_DIR}/rewards.csv", reward_history, delimiter=",")
    env.close()
    print("Training completed successfully.")


if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at {start_time}")
    train(env_name="CartPole-v1")  # replace with your own environment interface
