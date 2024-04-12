import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import math
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ActionValueFunctionApproximator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_state_space = 8
        n_action_space = 4
        hidden_space_1 = 128
        hidden_space_2 = 128

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_state_space, hidden_space_1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_space_1, hidden_space_2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_space_2, n_action_space)
        )

    def forward(self, x):
        return self.net(x.float())


class RLAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_value_function = ActionValueFunctionApproximator().to(self.device)
        self.target_value_function = ActionValueFunctionApproximator().to(self.device)
        self.target_value_function.load_state_dict(self.action_value_function.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.action_value_function.parameters(),
            lr=1e-4
        )
        
        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.996
        self.epsilon = 1.0
        self.batch_size = 128
        self.transitions = deque([], maxlen=self.batch_size)

    def getAction(self, env, s):
        if np.random.random() < self.epsilon:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            return self.action_value_function(torch.tensor(s, device=self.device, dtype=torch.float32)).unsqueeze(0).max(1).indices.view(1,1)

    def decayEpsilonAndTau(self):
        self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)
        self.tau = min(0.9, self.tau*(2-self.epsilon_decay))

    def update(self):
        batch = Transition(*zip(*random.sample(self.transitions, self.batch_size)))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        action_value_batch = self.action_value_function(state_batch).gather(1, action_batch)
        next_action_value_batch = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            next_action_value_batch[non_final_mask] = self.target_value_function(non_final_next_states).max(1).values
        sarsa_target_batch = reward_batch+self.gamma*next_action_value_batch
        
        criterion = torch.nn.MSELoss()
        loss = criterion(action_value_batch, sarsa_target_batch.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        target_net_state_dict = self.target_value_function.state_dict()
        current_net_state_dict = self.action_value_function.state_dict()
        for key in current_net_state_dict:
            target_net_state_dict[key] = self.tau*current_net_state_dict[key]+(1-self.tau)*target_net_state_dict[key]
        self.target_value_function.load_state_dict(target_net_state_dict)

        return loss.item()
    
def trainAgent(agent, num_episodes, env):
    rewards_over_episodes = []
    losses_over_updates = []
    average_rewards = []
    total_steps = 0
    for i in range(num_episodes):
        obs, info = env.reset()
        rewards_in_episode = []
        done = False
        while not done:
            action = agent.getAction(env, obs)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            rewards_in_episode.append(reward)
            done = terminated or truncated
            if done:
                next_obs = None
            agent.transitions.append(Transition(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0),
                action,
                torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0) if next_obs is not None else next_obs,
                torch.tensor([reward], dtype=torch.float32)
            ))
            obs = next_obs
            total_steps += 1
            if (total_steps%4)==0 and len(agent.transitions)>=agent.batch_size:
                losses_over_updates.append(agent.update())
        rewards_over_episodes.append(rewards_in_episode)
        agent.decayEpsilonAndTau()
        if i and i%100 == 0:
            print("Average loss in last 100 updates is: {}".format(np.mean(losses_over_updates[-100:])))
            rewards = 0
            for Rs in rewards_over_episodes[-100:]:
                rewards += np.sum(Rs)
            rewards /= 100
            average_rewards.append(rewards)
            print("Average Total reward for last 100 episodes is: {}".format(rewards))
    return rewards_over_episodes



def main():
    env = gym.make("LunarLander-v2", continuous=False)
    agent = RLAgent()
    num_episodes = 2500
    rewards_data = trainAgent(agent, num_episodes, env)
    env.close()
    torch.save(agent.action_value_function.state_dict(),"offPolicy_Sarsa0_QFunc.pth")
    env = gym.make("LunarLander-v2", render_mode="human", continuous=False)
    for _ in range(10):
        observation, info = env.reset()
        done = False
        while not done:
            action = agent.getAction(env, observation)
            observation, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
    env.close()

    total_reward_per_episode = [np.sum(episode_rewards) for episode_rewards in rewards_data]
    average_rewards = [(i,np.mean(total_reward_per_episode[i:i+100])) for i in range(0,num_episodes,100)]
    plt.figure()
    plt.plot(total_reward_per_episode)
    plt.plot(*zip(*average_rewards))
    plt.savefig("rewards_onPolicy_Sarsa0_QLearning.png")
    plt.show()
    return agent

def testModel(path):
    agent = RLAgent()
    agent.action_value_function.load_state_dict(torch.load(path))
    agent.epsilon = 0
    env = gym.make("LunarLander-v2", render_mode="human", continuous=False)
    for _ in range(10):
        observation, info = env.reset()
        done = False
        while not done:
            action = agent.getAction(env, observation)
            observation, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
    env.close()

if __name__ == "__main__":
    testModel("offPolicy_Sarsa0_QFunc1.pth")
    # main()