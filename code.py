"""
Deep Q-Network (DQN) on CartPole-v1 (Gymnasium)

CartPole is the classic inverted-pendulum / pole-balancing control problem:
a cart moves along a frictionless track, with a pole hinged on top of it.
The only control available is pushing the cart left or right, and the goal
is to keep the pole from falling over for as long as possible. This is a
real problem-shape used throughout robotics and control engineering (the
same underlying dynamics show up in Segway-style balancing robots, rocket
landing stabilisation, and industrial crane control), which is why it is
used here instead of a purely symbolic grid-world.

Why DQN (rather than tabular Q-Learning) for this problem?
CartPole's state space is CONTINUOUS: cart position, cart velocity, pole
angle, and pole angular velocity are all real-valued. A tabular Q-Learning
approach (as used for FrozenLake) is not viable here, because there is no
way to enumerate a finite table of states -- there are infinitely many
possible (position, velocity, angle, angular velocity) combinations. DQN
solves this by replacing the Q-table with a neural network Q(s, a; theta)
that generalises across similar, previously unseen states, which is
exactly the scalability problem DQN was designed to solve.

Key DQN mechanisms implemented below:
  1. A neural network function approximator (instead of a Q-table).
  2. Experience replay: past transitions are stored in a buffer and
     sampled in random mini-batches, which breaks the correlation between
     consecutive experiences and stabilises training.
  3. A target network: a slowly-updated copy of the Q-network used to
     compute stable TD targets, preventing the "moving target" problem
     that arises from bootstrapping off a constantly-changing network.
  4. Epsilon-greedy exploration with decay, as in the FrozenLake agent.
"""

import random
import copy
from collections import deque, namedtuple
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import json

# Environment setup
# CartPole-v1:
#   State (observation):  [cart position, cart velocity,
#                           pole angle,    pole angular velocity]  (4 continuous values)
#   Actions:               0 = push cart left, 1 = push cart right (2 discrete actions)
#   Reward:                +1 for every time step the pole stays upright
#   Episode ends when:     pole angle exceeds +/-12 degrees, cart leaves the
#                           +/-2.4 unit track, OR 500 steps are reached (a success)

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]   # 4
n_actions = env.action_space.n               # 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"State dim: {state_dim}, Actions: {n_actions}, Device: {device}")

# Q-Network (function approximator replacing the Q-table)
class QNetwork(nn.Module):
    """Maps a 4-dimensional state to Q-values for both discrete actions."""
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# Experience Replay Buffer
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# Hyperparameters
n_episodes = 500
max_steps_per_episode = 500  # CartPole-v1's own success cap
learning_rate = 5e-4
discount_factor = 0.99  # gamma
batch_size = 64
replay_capacity = 50000
min_replay_before_training = 1000
target_soft_update_tau = 0.005  # Polyak averaging coefficient for the target network
epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.98  # multiplicative decay per episode

# Initialise networks, optimizer, replay buffer
policy_net = QNetwork(state_dim, n_actions).to(device)
target_net = QNetwork(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())   # sync target net initially
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.SmoothL1Loss()   # Huber loss: more robust to outlier TD errors than MSE
replay_buffer = ReplayBuffer(replay_capacity)

def select_action(state, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return env.action_space.sample()           # explore
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())   # exploit

def optimize_model():
    # One gradient step of DQN training using a sampled mini-batch.
    if len(replay_buffer) < max(batch_size, min_replay_before_training):
        return None

    batch = replay_buffer.sample(batch_size)

    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    # Current Q-value estimates for the actions actually taken
    q_values = policy_net(states).gather(1, actions)

    # Bellman target using DOUBLE DQN: the policy network SELECTS the best
    # next action, but the target network EVALUATES it. This decouples
    # action selection from action evaluation, which corrects the
    # over-optimistic Q-value estimates that vanilla DQN is prone to
    # (a known cause of the training instability/collapse seen with a
    # single hard-synced target network on this problem).
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        max_next_q = target_net(next_states).gather(1, next_actions)
        td_target = rewards + discount_factor * max_next_q * (1 - dones)

    loss = loss_fn(q_values, td_target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)  # gradient clipping for stability
    optimizer.step()

    # Soft (Polyak) update of the target network after every training step,
    # instead of an infrequent hard copy. This keeps the TD target moving
    # slowly and smoothly rather than jumping abruptly every few episodes,
    # which further improves stability.
    with torch.no_grad():
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_soft_update_tau * policy_param.data
                + (1.0 - target_soft_update_tau) * target_param.data
            )

    return loss.item()

# Training loop
episode_rewards = []
losses = []

# Checkpointing: DQN can rediscover a good policy and then drift away from
# it again (a known instability sometimes called "catastrophic forgetting"
# in RL) if training continues on newly-collected, still partly exploratory
# data. To guard against ending training on a WORSE set of weights than one
# seen earlier, we keep a copy of the best-performing network snapshot and
# use that for testing/saving -- not simply whatever weights exist after
# the last episode.
best_avg_reward = -float("inf")
best_state_dict = copy.deepcopy(policy_net.state_dict())
solved_streak = 0

for episode in range(n_episodes):
    state, info = env.reset()
    total_reward = 0
    episode_loss = []

    for step in range(max_steps_per_episode):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, float(terminated))
        # NOTE: we bootstrap using `terminated` (not `truncated`) for the done flag,
        # since a truncated episode (hit the step cap) still has real future value,
        # while a terminated episode (pole fell) does not.

        state = next_state
        total_reward += reward

        loss = optimize_model()
        if loss is not None:
            episode_loss.append(loss)

        if done:
            break

    # (Target network is now updated via soft/Polyak averaging inside
    # optimize_model() after every gradient step, so no separate periodic
    # hard sync is needed here.)

    # Decay epsilon (exploration -> exploitation, as in the FrozenLake agent)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    episode_rewards.append(total_reward)
    losses.append(np.mean(episode_loss) if episode_loss else 0)

    # --- Checkpointing: keep the best snapshot seen so far ---
    avg_last_20 = np.mean(episode_rewards[-20:])
    if avg_last_20 > best_avg_reward:
        best_avg_reward = avg_last_20
        best_state_dict = copy.deepcopy(policy_net.state_dict())

    # --- Early stopping: once the policy is reliably solved, stop training.
    # This avoids exactly the failure mode of training past a good solution
    # and drifting into a worse one, rather than relying only on the
    # checkpoint as a safety net. ---
    if avg_last_20 >= 490 and episode >= 100:
        solved_streak += 1
    else:
        solved_streak = 0
    if solved_streak >= 10:
        print(f"Environment reliably solved (avg {avg_last_20:.1f} over last 20 episodes, "
              f"sustained for {solved_streak} episodes) -- stopping early at episode {episode+1}.")
        break

    if (episode + 1) % 20 == 0:
        print(f"Episode {episode+1:>4}/{n_episodes} | "
              f"avg reward (last 20): {avg_last_20:6.1f} | best avg so far: {best_avg_reward:6.1f} | "
              f"epsilon: {epsilon:.3f}")

env.close()

# Load the BEST checkpoint seen during training (not necessarily the final
# episode's weights) before evaluating and saving.
policy_net.load_state_dict(best_state_dict)
target_net.load_state_dict(best_state_dict)
print(f"\nUsing best checkpoint: avg reward {best_avg_reward:.1f} (20-episode moving average)")

# Performance evaluation
window = 20
moving_avg = [np.mean(episode_rewards[max(0, i - window):i + 1]) for i in range(len(episode_rewards))]

plt.figure(figsize=(8, 5))
plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
plt.plot(moving_avg, label=f"{window}-episode moving average", linewidth=2)
plt.axhline(y=475, color="green", linestyle="--", label="CartPole-v1 'solved' threshold (475)")
best_episode = int(np.argmax(moving_avg))
plt.scatter([best_episode], [moving_avg[best_episode]], color="red", zorder=5,
            label=f"Best checkpoint (episode {best_episode+1})")
plt.xlabel("Episode")
plt.ylabel("Total reward (= steps pole stayed upright)")
plt.title("DQN on CartPole-v1: Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_progress.png", dpi=150)
print("Saved training_progress.png")

# Testing the trained agent (greedy policy, epsilon=0)
test_env = gym.make("CartPole-v1")
n_test_episodes = 100
test_rewards = []

policy_net.eval()
for _ in range(n_test_episodes):
    state, info = test_env.reset()
    total_reward = 0
    for step in range(max_steps_per_episode):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(torch.argmax(policy_net(state_t), dim=1).item())
        state, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    test_rewards.append(total_reward)

test_env.close()

avg_test_reward = float(np.mean(test_rewards))
solved_fraction = float(np.mean([r >= 475 for r in test_rewards]))

print(f"\nTest results over {n_test_episodes} episodes:")
print(f"  Average reward (avg steps balanced): {avg_test_reward:.1f} / 500")
print(f"  Fraction of episodes 'solved' (>=475 steps): {solved_fraction:.1%}")

# Save results for the report
results = {
    "n_episodes": n_episodes,
    "episodes_run": len(episode_rewards),
    "learning_rate": learning_rate,
    "discount_factor": discount_factor,
    "batch_size": batch_size,
    "target_soft_update_tau": target_soft_update_tau,
    "epsilon_decay": epsilon_decay,
    "min_epsilon": min_epsilon,
    "episode_rewards": episode_rewards,
    "best_checkpoint_avg_reward": float(best_avg_reward),
    "avg_test_reward": avg_test_reward,
    "solved_fraction": solved_fraction,
    "test_rewards": test_rewards,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved results.json")

torch.save(policy_net.state_dict(), "dqn_cartpole_weights.pt")
print("Saved dqn_cartpole_weights.pt")
