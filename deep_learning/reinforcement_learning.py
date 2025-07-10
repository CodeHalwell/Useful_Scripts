"""Basic reinforcement learning helpers using Q-learning."""

import numpy as np
import torch
from torch import nn


def epsilon_greedy(q_values: np.ndarray, epsilon: float) -> int:
    """Select an action using epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    return int(np.argmax(q_values))


def q_learning(env, episodes: int = 1000, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
    """Run a simple Q-learning loop for a Gym-like environment."""
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = epsilon_greedy(q_table[state], epsilon)
            next_state, reward, done, _ = env.step(action)
            best_next = np.max(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * best_next - q_table[state, action])
            state = next_state
    return q_table


class DQN(nn.Module):
    """A minimal Deep Q-Network."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def dqn_learning(env, episodes: int = 500, gamma: float = 0.99, epsilon: float = 0.1, lr: float = 1e-3):
    """Train a simple DQN on ``env`` and return the trained network."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_net = DQN(state_dim, action_dim)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(episodes):
        state = env.reset()
        done = False
        state = torch.tensor(state, dtype=torch.float32)
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(torch.argmax(q_net(state)).item())
            next_state, reward, done, _ = env.step(action)
            next_state_t = torch.tensor(next_state, dtype=torch.float32)
            target = reward + gamma * (0.0 if done else torch.max(q_net(next_state_t)).item())
            q_value = q_net(state)[action]
            loss = loss_fn(q_value, torch.tensor(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state_t
    return q_net
