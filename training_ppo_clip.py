import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from matplotlib import pyplot as plt
from torch.distributions import Categorical


# Set the seeds
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)



class Actor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Actor, self).__init__()    
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
    
    def forward(self, state):
        activation1 = F.relu(self.layer1(state))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2) 
        return output

    
class Critic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Critic, self).__init__()    
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, out_dim)

    def forward(self, state):
        activation1 = F.relu(self.layer1(state))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)  
        return output


class PPO:
    def __init__(self, env, eval_env):
        # Extract environment information
        self.env = env
        self.eval_env = eval_env
        self.state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.action_size = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE: {self.device}")

        self._init_hyperparameters()

        # Initialize actor and critic networks
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size, 1).to(self.device)

        # Initialize optimizers for actor and critic
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    
    def _init_hyperparameters(self):
        # Default values for hyperparameters
        self.eps_per_batch = 100                   # episodes per batch
        self.max_timesteps_per_episode = 40        # timesteps per episode
        self.n_updates_per_iteration = 4           # number of times to update actor/critic per iteration
        self.gamma = 0.99                          # discount factor
        #self.lam = 0.99                            # lambda parameter
        self.lr_actor = 3e-4                       # learning rate of actor optimizer
        self.lr_critic = 5e-4                      # learning rate of critic optimizer
        self.clip = 0.2                            # clip factor
        self.ent_coef = 0.01                       # entropy coefficient for entropy regularization
        self.target_kl = 0.02                      # KL Divergence threshold
        self.max_grad_norm = 0.5                   # max gradient norm

    def learn(self, total_eps):
        '''Learning function'''
        t_so_far = 0
        evals = [] # Contains the average returns of the evaluations performed during training
        eps = 0
        total_rewards = [] # Contains the returns of episodes during training
        i = 0 # Batch number

        while eps < total_eps:
            # Collecting our batch simulations
            i += 1 # Increment batch number
            print("\n")
            print(f"BATCH {i} \n")
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_sum_rews= self.rollout() 
            total_rewards.extend(batch_sum_rews)

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            eps += len(batch_lens)
            

            # Calculate advantage at k-th iteration
            #A_k = self.calculate_gae(batch_rews, batch_values, batch_dones)

            # Calculate value at k-th iteration
            V_k = self.critic(batch_obs).squeeze()
            
            # Calculate advantage at k-th iteration
            A_k = batch_rtgs - V_k.detach()  
            
            
            #V, _, _ = self.evaluate(batch_obs, batch_acts)
            #A_k = batch_rtgs - V.detach() 

            # Advantage normalization
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            

            actor_losses = []
            critic_losses = []

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):

                
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                entropy_loss = entropy.mean()
                actor_loss = actor_loss - self.ent_coef * entropy_loss
                
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                logratios = curr_log_probs - batch_log_probs
                ratios = torch.exp(logratios)
                approx_kl = ((ratios - 1) - logratios).mean()

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()

                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # Gradient clip
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    print(f"KL Divergence target exceeded after {t_so_far} timesteps")
                    break # if KL above threshold

                actor_losses.append(actor_loss.detach().cpu().numpy())
                critic_losses.append(critic_loss.detach().cpu().numpy())

            # Evaluation during training
            print("\n")
            print("EVALUATION DURING TRAINING")
            evals.append(self.test(eval_env))

        
        return evals, total_rewards

    def rollout(self):
        '''Create a batch of experience'''
        # Batch data
        batch_states = []             # batch observations
        batch_actions = []            # batch actions
        batch_log_probs = []          # log probs of each action
        batch_rews = []               # batch rewards
        #batch_values = []
        batch_lens = []               # episodic lengths in batch
        batch_dones = []

        batch_sum_rews = []

        t = 0 # Keeps track of how many timesteps we have run so far this batch
        eps = 0
        while eps < self.eps_per_batch:
            ep_rews = [] # rewards collected per episode
            #ep_values = []
            ep_dones = []

			# Reset the environment 
            state, _ = self.env.reset()
            done = False
            eps += 1
            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for t_ep in range(self.max_timesteps_per_episode):
                
                ep_dones.append(done)
                t += 1 # Increment timesteps ran this batch so far
                
                # Track observations in this batch
                state = state.reshape(-1)
                batch_states.append(state)

                # Calculate action and make a step in the environment 
                action, log_prob = self.get_action(state)
                #value = self.critic(obs)

                state, reward, terminated, truncated, _ = self.env.step(action)

                # Don't really care about the difference between terminated or truncated, so just combine them
                done = terminated | truncated

                # Track recent reward, action, and action log probability
                ep_rews.append(reward)
                #ep_values.append(value.flatten())
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    print(f"Total T: {t}, Episode Num: {eps}, Episode T: {t_ep}, Return: {sum(ep_rews):.3f}")
                    break
            
            # Track episodic lengths and rewards
            batch_sum_rews.append(sum(ep_rews))
            batch_lens.append(t_ep + 1)
            batch_rews.append(ep_rews)
            #batch_values.append(ep_values)
            batch_dones.append(ep_dones)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float).to(self.device)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
        batch_rtgs = self.compute_rtgs(batch_rews)
    
        #return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_values, batch_dones, batch_sum_rews
        return batch_states, batch_actions, batch_log_probs, batch_rtgs, batch_lens, batch_sum_rews


    def get_action(self, state):
        """
			Queries an action from the actor network.

			Parameters:
				state - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
        # Get action probability distribution
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().detach().numpy(), log_prob.detach()
    
    def get_best_action(self, state):
        """
			Queries the best action from the actor network.

			Parameters:
				state - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs = self.actor(state)
        action = torch.argmax(action_probs).item()
        print(action)
        return action
    
    
    def compute_rtgs(self, batch_rewards):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rewards):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode
            # Discounted return
            for reward in reversed(ep_rews):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

        return batch_rtgs
    
    def calculate_gae(self, batch_rewards, batch_values, batch_dones):
        # NOTE: this function is not actually used in the code.
        # I tried to use GAE as an alternative approach
        """
            Calculate advantages using generalized advantage estimation.
            Parameters:
                batch_rewards - the rewards from the most recently collected batch as a tensor.
                batch_values - the values from the most recently collected batch as a tensor.
                batch_dones - the dones from the most recently collected batch as a tensor.
                            
            Return:
                batch_advatages - the advantages from the most recently collected batch as a tensor.
                
        """
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(batch_rewards, batch_values, batch_dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float).to(self.device)
    
    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
                dist.entropy() - entropy of the distribution
        """
        # Query critic network for a value V for each batch_obs
        V = self.critic(batch_obs).squeeze().to(self.device)

        # Calculate the log probabilities of batch actions using most recent actor network.
        action_probs = self.actor(batch_obs)
        dist = Categorical(logits=action_probs)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch,
        # log probabilities log_probs of each action in the batch
        # and entropy of the distribution
        return V, log_probs, dist.entropy()

    def test(self, env, episodes=10):
        """
        Evaluates the agent on the environment for a given number of episodes.
        """
        total_cumulative_reward = 0
        for episode in range(episodes):
            state, _ = env.reset()
            episode_cumulative_reward = 0
            t = 0
            episode_rewards = []
            while True:
                state = state.reshape(-1)
                action, _ = agent.get_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_rewards.append(reward)
                episode_cumulative_reward += reward # Sum of rewards
                t += 1
                if terminated | truncated:
                    print(f"Episode Num: {episode}, Episode T: {t}, Return: {episode_cumulative_reward:.3f}")
                    break
            total_cumulative_reward += episode_cumulative_reward

        average_cumulative_reward = total_cumulative_reward / episodes # Average sum of rewards
        print(f"Average Sum of Rewards during Evaluation: {average_cumulative_reward}")
        print("\n")
        return average_cumulative_reward

    

if __name__ == "__main__":
    env_name = "highway-fast-v0"  # environment for faster training"
    eval_env_name = "highway-v0"

    env = gymnasium.make(env_name,
                        config={'action': {'type': 'DiscreteMetaAction'}, 'policy_frequency' : 5, 'duration': 120, "vehicles_count": 50, "lanes_count": 3},
                        render_mode=None)                
            


    eval_env = gymnasium.make(eval_env_name,
                        config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                        render_mode='human')

    env = env.unwrapped
    print(env.config)
    
    # Agent
    agent = PPO(env, eval_env)

    num_episodes = 3000 # Number of episodes in training

    # Train the agent
    evals, total_rewards = agent.learn(num_episodes)


    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Sum of Rewards")
    plt.title("PPO training")
    #plt.savefig("training_ppo.pdf") # Save plot
    plt.show()

    episodes = np.arange(100, 3100, 100)
    plt.figure()
    plt.plot(episodes, evals)
    plt.xlabel("Episodes")
    plt.ylabel("Average Sum of Rewards")
    plt.title("PPO evaluation during training")
    #plt.savefig("eval_ppo_during_training.pdf") # Save plot
    plt.show()

    env.close()
