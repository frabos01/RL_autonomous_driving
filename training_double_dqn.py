import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from matplotlib import pyplot as plt
import math
from collections import deque


# Set the seeds
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


class DoubleDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DoubleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        if(state.dim() == 3):
            state = state.contiguous().view(state.size(0), -1)  # Flatten to (batch_size, 5*5)
        
        else:
            state = state.contiguous().view(-1) # Flatten to (5*5)
        return self.network(state)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        '''Push experience in the replay buffer.'''
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''Sample a batch of experiences from the replay buffer'''
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))

    def __len__(self):
        '''Length of the replay buffer'''
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, eps_max=1.0, eps_min=0.05, lr=1e-3, buffer_size=10000, tau=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma # Discounted factor
        self.batch_size = 64
        self.tau = tau # Tau for soft update

        # Epsilons for epsilon decay
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.epsilon = self.eps_max
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE: {self.device}")

        self.q_network = DoubleDQN(state_size, action_size).to(self.device)
        self.target_network = DoubleDQN(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() #  Evaluation mode

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)
    

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_network(state).argmax().item()
    
    def select_best_action(self, state):
        '''Greedy action selection'''
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def train(self, timestep, total_timesteps):
        '''Perform a training step'''

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values
        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        # Update the Q-network
        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target network
        self.update_target_network()

       # Linear epsilon decay
        self.epsilon = max(self.eps_min, self.eps_max - (timestep / total_timesteps) * (self.eps_max - self.eps_min))

    def update_target_network(self):
        '''Soft update the target network'''
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
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

    env_name = "highway-fast-v0"  # Env for training
    eval_env_name = "highway-v0"  # Env for evaluation

    env = gymnasium.make(env_name,
                        config={'action': {'type': 'DiscreteMetaAction'}, 'duration': 40, "vehicles_count": 50, "lanes_count": 3},
                        render_mode=None)                
            


    eval_env = gymnasium.make(eval_env_name,
                        config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                        render_mode='human')

    env = env.unwrapped
    eval_env = eval_env.unwrapped

    print(env.config)
    
 
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    # Agent
    agent = DQNAgent(state_size, action_size)

    num_episodes = 3000 # Total number of episodes
    t = 0 # Initialize global timestep
    total_t = 2e4 # Set T for beta scheduling and epsilon decay (this is not the total number of timesteps)
    evals = [] # Contains the average returns of the evaluations performed during training
    total_rewards = [] # Contains the returns of episodes during training

    # Training
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        t_ep = 0 # Initialize episode timestep
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            # Add experience in the replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            t += 1 # Increment global timestep
            t_ep += 1 # Increment episode timestep
            
            # Perform a training step
            agent.train(t, total_t)

        print(f"Episode {episode + 1}, Episode T: {t_ep}, Return: {total_reward:.3f}, Epsilon: {agent.epsilon:.3f}")
        total_rewards.append(total_reward)

        # Evaluation during training
        if((episode+1) % 100 == 0):
            print("\n")
            print("EVALUATION DURING TRAINING")
            average_total_reward = agent.test(eval_env)
            evals.append(average_total_reward)

            # Save model
            '''if(episode > 400):
                checkpoint_path = f"q_network_episode_{episode+1}_double_dqn.pth"
                torch.save(agent.q_network.state_dict(), checkpoint_path)

                checkpoint_path = f"target_network_episode_{episode+1}_double_dqn.pth"
                torch.save(agent.target_network.state_dict(), checkpoint_path)'''
    

    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Double DQN Training")
    #plt.savefig("training_double_dqn.pdf") # Save plot
    plt.show()

    episodes = np.arange(100, 3100, 100)
    plt.figure()
    plt.plot(episodes, evals)
    plt.xlabel("Episodes")
    plt.ylabel("Average Sum of Rewards")
    plt.title("Double DQN evaluation during training")
    #plt.savefig("eval_double_dqn_during_training.pdf") # Save plot
    plt.show()

    env.close()
