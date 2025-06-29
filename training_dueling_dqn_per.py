import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from matplotlib import pyplot as plt


# Set the seeds
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Common feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, state):
        if(state.dim() == 3):
            state = state.contiguous().view(state.size(0), -1)  # Flatten to (batch_size, 5*5)
        
        else:
            state = state.contiguous().view(-1) # Flatten to (5*5)

        features = self.feature_layer(state)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine them into Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class SumTree:
    """Binary tree where parent nodes store the sum of child priorities."""
    def __init__(self, capacity):
        self.capacity = capacity # Capacity of the experience buffer
        self.tree = np.zeros(2 * capacity - 1)  # Sum tree array
        self.data = np.zeros(capacity, dtype=object)  # Experience buffer
        self.write = 0 # Tracks where to write new experiences in the buffer
        self.total = 0 # Counts the total number of experiences added

    def add(self, priority, data):
        """Add a new experience with priority."""
        idx = self.write + self.capacity - 1  # Find leaf index in the tree, note that leaves are in the last $self.capacity indices
        self.data[self.write] = data  # Store experience
        self.update(idx, priority)  # Update tree of priorities
        self.write = (self.write + 1) % self.capacity  # Circular buffer
        self.total += 1

    def update(self, idx, priority):
        """
            Update priority and propagate changes up the tree.
            Parameters:
				priority - the priority of an experience
                idx - index associated to the priority in the Sum Tree
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, priority):
        """
            Sample experience based on priority.
            Parameters:
				priority - the priority of an experience

			Return:
				idx - index associated to the priority in the Sum Tree
				self.tree[idx] - the priority in the Sum Tree
                self.data[data_idx] - data in the experience buffer

        """
        idx = 0 # Initialize index associated to the priority in the Sum Tree
        # Traverse Sum Tree
        while idx < self.capacity - 1:
            left = 2 * idx + 1 # Index of the left child node
            right = left + 1 # Index of the right child node
            if priority <= self.tree[left]: # Go left
                idx = left
            else: # Go right
                priority -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1 # Index of the data in the experience buffer
        return idx, self.tree[idx], self.data[data_idx]

    def total_priority(self):
        """Return the root node value (sum of all priorities)."""
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, batch_size, alpha=0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha  # Priority exponent
        self.epsilon = 1e-5  # Small value to prevent zero priority

    def add(self, state, action, reward, next_state, done):
        """Add experience with maximum priority."""
        max_priority = max(self.tree.tree[-self.tree.capacity:]) if self.tree.total > 0 else 1.0
        self.tree.add(max_priority, (state, action, reward, next_state, done))

    def sample(self, beta=0.4):
        """Sample a batch of experiences using priority-based sampling."""
        batch = []
        idxs = []
        # Total priority of the Sum Tree is divided into equal segments
        segment = self.tree.total_priority() / self.batch_size # Segment legth
        priorities = []

        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            # For each sample, a random value is generated within its segment
            s = np.random.uniform(a, b)
            # Higher priority experiences occupy larger segments in the tree, thus have a higher chance of being selected
            idx, priority, data = self.tree.get_leaf(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # Sampling probabilities
        sampling_probs = priorities / self.tree.total_priority()

        # The importance sampling weights
        weights = (self.capacity * sampling_probs) ** (-beta)
        weights /= weights.max()  # Normalization

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones), idxs, np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, idxs, td_errors):
        """Update priorities based on TD errors."""
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)

class DuelingDQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, tau=1e-3,
                 eps_max=1.0, eps_min=0.05, alpha=0.6, beta_start=0.4, beta_final=1.0):
        
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma # Discounted factor
        self.tau = tau # Tau for soft upgrade
        self.lr = lr # Learning rate

        # Epsilons for epsilon decay
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.epsilon = self.eps_max

        # PER parameters
        self.alpha = alpha # Controls the degree of prioritization
        self.beta = beta_start  # Controls the amount of importance sampling correction
        self.beta_start = beta_start
        self.beta_final = beta_final

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE: {self.device}")

        # Networks
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, batch_size, alpha)

        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_target_network(self):
        """Soft update the target network."""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def select_best_action(self, state):
        '''Greedy action selection'''
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience (S, A, R, S', D) in the buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self, timestep, total_timesteps):
        '''Perform a training step'''
        if self.replay_buffer.tree.total < self.batch_size:
            return
        
        # Sample experiences from the buffer
        states, actions, rewards, next_states, dones, idxs, weights = self.replay_buffer.sample(self.beta)
        
        # Beta scheduling: linear increase from beta_start to beta_final
        self.beta = self.beta_start + (self.beta_final - self.beta_start) * (timestep / total_timesteps)
        self.beta = min(self.beta_final, self.beta)  # Ensure beta never exceeds 1.0

        # Linear epsilon decay
        self.epsilon = max(self.eps_min, self.eps_max - (timestep / total_timesteps) * (self.eps_max - self.eps_min))

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Calculate Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Calculate TD errors
        td_errors = target_q_values - current_q_values

        # Calculate loss and perform backpropagation
        loss = (weights * td_errors.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update prorities in the buffer
        self.replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())

        # Apply soft update to target network
        self.update_target_network()
    
    def test(self, eval_env):
        '''Evaluate the model during training'''
        total_reward = 0
        for ep in range(10):
            state, _ = eval_env.reset()
            episode_total_reward = 0
            t_ep = 0
            while True:
                action = agent.select_action(state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                t_ep += 1
                episode_total_reward += reward
                if terminated | truncated:
                    print(f"Episode Num: {ep+1}, Episode T: {t_ep}, Return: {episode_total_reward:.3f}")
                    break
            total_reward += episode_total_reward

        average_total_reward = total_reward / 10
        print(f"Average Return during Evaluation: {average_total_reward:.3f}")
        print("\n")
        return average_total_reward



if __name__ == "__main__":

    env_name = "highway-fast-v0" # Env for training
    eval_env_name = "highway-v0" # Env for evaluation

    env = gymnasium.make(env_name,
                        config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "vehicles_count": 50, "lanes_count": 3},
                        render_mode=None)                
            


    eval_env = gymnasium.make(eval_env_name,
                        config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                        render_mode='human')

    env = env.unwrapped
    print(env.config)

    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    # Agent
    agent = DuelingDQNAgent(state_size, action_size)

    num_episodes = 4000 # Total number of episodes
    total_t = 2e4  # Set T for beta scheduling and epsilon decay (this is not the total number of timesteps)
    t = 0  # Initialize global timestep
    total_rewards = [] # Contains the returns of episodes during training
    evals = [] # Contains the average returns of the evaluations performed during training

    # Training
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        t_ep = 0 # Initialize episode timestep
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            t += 1 # Increment global timestep
            t_ep += 1 # Increment episode timestep

            # Add experience in the replay buffer
            agent.add_experience(state, action, reward, next_state, done)

            # Perform a training step
            agent.train(t, total_t)
            
            state = next_state
            total_reward += reward
            
        
        print(f"Episode {episode + 1}, Episode T: {t_ep}, Return: {total_reward:.3f}, Epsilon: {agent.epsilon:.3f}, Beta: {agent.beta:.3f}")
        total_rewards.append(total_reward)

        # Evaluation during training
        if((episode+1) % 100 == 0):
            print("\n")
            print("EVALUATION DURING TRAINING")
            average_total_reward = agent.test(eval_env)
            evals.append(average_total_reward)

            # Save model
            if(episode > 400):
                checkpoint_path = f"q_network_episode_{episode+1}_per.pth"
                torch.save(agent.q_network.state_dict(), checkpoint_path)

                checkpoint_path = f"target_network_episode_{episode+1}_per.pth"
                torch.save(agent.target_network.state_dict(), checkpoint_path)

    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Dueling DQN PER Training")
    #plt.savefig("training_dueling_dqn_per.pdf") # Save plot
    plt.show()

    episodes = np.arange(100, 4100, 100)
    plt.figure()
    plt.plot(episodes, evals)
    plt.xlabel("Episodes")
    plt.ylabel("Average Sum of Rewards")
    plt.title("Dueling DQN PER evaluation during training")
    #plt.savefig("eval_dueling_dqn_per_during_training.pdf") # Save plot
    plt.show()

    env.close()
    
