# reinforcement_learning.py

The reinforcement learning helpers demonstrate both tabular Q-learning and a minimal Deep Q-Network (DQN) using
PyTorch. Reinforcement learning optimizes an agent's behaviour through trial and error interactions with an environment.

## Functions and Classes

- **`epsilon_greedy`** – selects an action from Q-values using an exploration parameter `epsilon`.
- **`q_learning`** – implements classic Q-learning for environments with discrete states. It maintains a Q-table and
  updates it using the Bellman equation.
- **`DQN`** – a neural network that approximates the action-value function for environments with continuous states.
- **`dqn_learning`** – trains the DQN on an environment using an epsilon-greedy policy and mean-squared error loss.

## Python Notes

The training loops rely on NumPy for array manipulations and PyTorch for tensor operations and optimization. The
functions show how to interact with Gym-like environments by calling `env.reset()` and `env.step(action)`.

## Theory

Q-learning iteratively updates Q-values using the reward plus the discounted value of the best next state. Deep
Q-Networks replace the lookup table with a neural network to handle large or continuous state spaces. Exploration via
epsilon-greedy ensures the agent occasionally tries new actions. The combination of gradient-based updates and reward
signals enables agents to learn control policies from experience.

Viewed through the lens of dynamic programming, reinforcement learning seeks to
approximate the optimal action-value function. Tabular methods update a lookup
table directly, while DQNs parameterize this function with a neural network so
that high-dimensional or continuous states can be tackled. The use of a discount
factor encodes a preference for immediate rewards, and the exploration schedule
balances gathering new experience versus exploiting current knowledge.
