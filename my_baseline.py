import gymnasium
import highway_env
import numpy as np
import math
from matplotlib import pyplot as plt

# Actions
LEFT = 0
IDLE = 1
RIGHT = 2
FASTER = 3
SLOWER = 4

# Tolerance in measurement
EPS = 1e-5

def select_action(state, episode_steps):
    action = None
    # This ensures thet the car goes to the right lane
    # NOTE: if the car starts from the left lane two actions 'RIGHT' are needed
    if episode_steps < 2:
        action = RIGHT
    else:
        distance = 0.16
        # Check if there is a car within a certain distance on x axis in the right lane
        any_car_within_distance = lambda state: np.any((state[1:, 1] > 0) & (state[1:, 1] < distance) & (abs(state[1:, 2]) < EPS))
        if(any_car_within_distance(state)):
            action = SLOWER
        else:
            action = FASTER
    return action


if __name__ == "__main__":

    env_name = "highway-v0"

    env = gymnasium.make(env_name,
                        config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                        render_mode='human')

    env = env.unwrapped
    print(env.config)

    num_episodes = 10
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_steps = 0  # Initialize timesteps
        done = False
        while not done:
            action = select_action(state, episode_steps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            episode_steps += 1       
            state = next_state
            total_reward += reward
            
        
        print(f"Episode {episode + 1}, Episode T: {episode_steps}, Total Reward: {total_reward:.3f}")
        total_rewards.append(total_reward)

    episodes = np.arange(1, num_episodes+1)
    average_total_reward = np.mean(total_rewards)

    print(f"Average Total Reward: {average_total_reward:.3f}")

    env.close()

    plt.figure()
    plt.plot(episodes, total_rewards)
    plt.plot(episodes, [average_total_reward] * num_episodes, color='r')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Baseline")
    #plt.savefig("baseline.pdf") # Save plot
    plt.show()

    
