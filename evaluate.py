import gymnasium
import highway_env
import numpy as np
import torch
import random
from matplotlib import pyplot as plt
from training_dueling_dqn_per import DuelingDQNAgent


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

def evaluate_model(env, agent, episodes=10):
    """
    Evaluates the agent on the environment for a given number of episodes.
    """
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        episode_total_reward = 0
        t = 0
        while True:
            action = agent.select_best_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_total_reward += reward
            t += 1
            if terminated | truncated:
                print(f"EPISODE {ep+1}, EPISODE T {t}, TOTAL REWARD: {episode_total_reward:.3f}")
                break
        total_rewards.append(episode_total_reward)

    average_total_reward = np.mean(total_rewards)
    print(f"Average Total Reward during Evaluation: {average_total_reward:.3f}")
    return total_rewards, average_total_reward



env_name = "highway-v0"

env = gymnasium.make(env_name,
                     config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n

agent = DuelingDQNAgent(state_size, action_size)

# Load best model
agent.q_network.load_state_dict(torch.load("dueling_q_network.pth", map_location=device, weights_only=True)) # Load q-network
agent.target_network.load_state_dict(torch.load("dueling_target_network.pth", map_location=device, weights_only=True)) # Load target network

num_episodes = 10
total_rewards, average_total_reward = evaluate_model(env, agent, num_episodes) # Model evaluation

env.close()

episodes = np.arange(1, num_episodes+1)
plt.figure()
plt.plot(episodes, total_rewards)
plt.plot(episodes, [average_total_reward] * num_episodes, color='r')
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Best Model Evaluation")
#plt.savefig("best_model_evaluation.pdf") # Save plot
plt.show()