import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import csv
# Ensure the model directory exists
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
    
    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.8):
        self.alpha = alpha  # controls the amount of prioritization used
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)  # Ensure state is correctly shaped as (1, num_features)
        next_state = np.expand_dims(next_state, 0)  # Same for next_state
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.max_priority
        self.tree.add(max_priority, (state, action, reward, next_state, done))
    
    def sample(self, batch_size, beta):
        batch = []
        indices = np.empty(batch_size, dtype=int)
        priorities = np.empty(batch_size, dtype=float)
        segment = self.tree.tree[0] / batch_size
        priorities_sum = self.tree.tree[0]
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            batch.append(data)
            indices[i] = index
            priorities[i] = priority
        sample_probs = priorities / priorities_sum
        weights = (self.capacity * sample_probs) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.concatenate(states), np.array(actions), np.array(rewards), np.concatenate(next_states), np.array(dones), indices, weights
    
    def update_priorities(self, indices, errors, offset=0.1):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + offset) ** self.alpha
            self.tree.update(idx, priority)
    
    def __len__(self):
        return len(self.tree.data)

# Define the DQN model
class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output a single value V(s)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)  # Output the advantage for each action A(s, a)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Recombine value and advantages to get final Q-values
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
def choose_action(state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            # Convert state to torch tensor, ensure it is a sequence (array)
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
            q_values = model(state)
            return q_values.max(1)[1].item()

def compute_loss(batch_size, beta=0.7):
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta=beta)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.LongTensor(action).unsqueeze(-1).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)
    weights = torch.FloatTensor(weights).to(device)

    # Get the Q-values for current state using the online model
    q_value = model(state).gather(1, action).squeeze(-1)

    # Double DQN part: select best action in next state using the online model, evaluate it using the target model
    best_action = model(next_state).max(1)[1].unsqueeze(1)
    next_q_state_values = target_model(next_state).gather(1, best_action).squeeze(1).detach()

    # Compute the target Q-value
    expected_q_value = reward + gamma * next_q_state_values * (1 - done)

    # Compute the weighted MSE loss
    loss = nn.functional.smooth_l1_loss(q_value, expected_q_value, reduction='none')
    loss = (loss * weights).mean()  # Apply weights and average

    # Calculate priorities based on the loss, adding a small constant to ensure no priority is zero
    prios = loss.detach().item() + 1e-5  # Ensure it's a scalar
    prios = np.full(len(indices), prios)  # Create a full array of this priority for all sampled indices
    replay_buffer.update_priorities(indices, prios)
    return loss

# Simulate the trained agent
def simulate_agent(env, model, num_episodes=5):
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        while True:
            action = choose_action(state, 0, num_actions)  # Choose best action (epsilon=0)
            state, reward, done, _, _ = env.step(action)
            state = state if isinstance(state, np.ndarray) else np.array([state])
            episode_reward += reward
            if done:
                break
        print(f"Simulated Episode {episode+1}: Total Reward: {episode_reward}")

# Plotting the total rewards
def plot_rewards(episode_rewards):
    plt.plot(episode_rewards)
    plt.title('Total Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# Hyperparameters
batch_size = 128
gamma = 0.99
replay_buffer_size = 100000
learning_rate = 1e-3
num_episodes = 1000
target_update = 10
epsilon = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995

#epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * frame_idx / epsilon_decay)

# Environment setup
env = gym.make('LunarLander-v2')
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

# Model and Optimizer
model = DuelingDQN(num_inputs, num_actions).to(device)
target_model = DuelingDQN(num_inputs, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning Rate Scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.99)

# Initialize replay buffer
replay_buffer = PrioritizedReplayBuffer(replay_buffer_size)

# Initialize target network with same weights
target_model.load_state_dict(model.state_dict())

episode_rewards = []
learn_rate = 4

# Training Loop
start_time = time.time()
for episode in range(num_episodes):
    state = env.reset()[0]  # Extract the state array assuming env.reset() returns a tuple (state, {})
    episode_reward = 0
    done = False
    truncated = False
    step = 0

    for step in range(1000):  # max steps per episode
        #epsilon = epsilon_by_frame(episode * 1000 + step)
        action = choose_action(state, epsilon, num_actions)
        next_state, reward, done, _, _ = env.step(action)
        
        # Ensure next_state is an array
        next_state = next_state if isinstance(next_state, np.ndarray) else np.array([next_state])

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state  # State for the next step is the current next_state

        episode_reward += reward

        if len(replay_buffer) > batch_size and step > 0 and step % learn_rate == 0:
            loss = compute_loss(batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

        #step += 1
    
    # Update the learning rate
    #scheduler.step()
    
    if episode % target_update == 0:
        target_model.load_state_dict(model.state_dict())

    if episode % 200 == 0:
        model_dir = 'Reward'
        os.makedirs(model_dir, exist_ok=True)
        filename = os.path.join(model_dir, f'DuelingDQN_epi_rewards_at_ep{episode}.csv')
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for reward in episode_rewards:
                writer.writerow([reward])
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Episode: {episode}, Total reward: {episode_reward}, Epsilon: {epsilon:.2f}, Learning_rate:{current_lr}')
    
    if epsilon > epsilon_final:
        epsilon *= epsilon_decay
    else:
        epsilon = epsilon_final
    
    episode_rewards.append(episode_reward)

end_time = time.time()

# Calculate the total time taken for training
total_training_time = end_time - start_time
hours, rem = divmod(total_training_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total training time: {int(hours)}:{int(minutes):02d}:{seconds:05.2f}")

# Save the model again after simulation, if it was further improved
torch.save(model.state_dict(), os.path.join(model_dir, 'lunar_lander_DuelingDQN_lr_1e-3.pth'))

# Plot the rewards
print("Plotting the rewards over episodes...")
N = 100
avg_reward = []
for i in range(len(episode_rewards)):
    if i < N:
        avg_reward.append(np.mean(episode_rewards[:i+1]))
    else:
        avg_reward.append(np.mean(episode_rewards[i-N+1:i+1]))

plot_rewards(avg_reward)

# Save the average rewards as csv
model_dir = 'Reward'
os.makedirs(model_dir, exist_ok=True)
filename = os.path.join(model_dir, 'DuelingDQN_average_rewards.csv')
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    for reward in avg_reward:
        writer.writerow([reward])

env.close()

# Create a new environment for simulation with rendering
env = gym.make('LunarLander-v2', render_mode="human")

print("Starting simulation of the trained agent...")

# Simulate the trained agent
simulate_agent(env, model)

# Close the simulation environment
env.close()
