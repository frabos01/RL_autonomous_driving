import gymnasium
import highway_env
import numpy as np
from matplotlib import pyplot as plt


env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={"manual_control": True, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

env.reset()
terminated, truncated = False, False

num_episodes = 10
episode = 1
episode_steps = 0
episode_return = 0
returns  = []
while episode <= num_episodes:
    episode_steps += 1

    _, reward, terminated, truncated, _ = env.step(env.action_space.sample())  # With manual control these actions are ignored
    env.render()

    episode_return += reward

    if terminated or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {terminated}")
        returns.append(episode_return)
        env.reset()
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

episodes = np.arange(1, num_episodes+1)
average_return = sum(returns) / num_episodes
plt.figure()
plt.plot(episodes, returns)
plt.plot(episodes, [average_return] * num_episodes, color='r')
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Manual Control")
#plt.savefig("manual_control.pdf") # Save plot
plt.show()