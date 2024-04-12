import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple

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
        self.optimizer = torch.optim.AdamW(
            self.action_value_function.parameters(),
            lr=1e-4
        )
        self.gamma = 0.99
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.996/25
        self.epsilon = 1.0
        self.replay_memory = deque([], maxlen=1000)
        self.batch_size = 64
        # self.rewards_actual = []
        # self.rewards_estimated = []
        # self.dfs = []

    def getAction(self, env, s):
        if np.random.random() < self.epsilon:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            return self.action_value_function(torch.tensor(s, device=self.device, dtype=torch.float32)).unsqueeze(0).max(1).indices.view(1,1)

    def decayEpsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon*self.epsilon_decay)
        
    def update(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, targets = zip(*batch)
        states = torch.cat(states)
        actions = torch.cat(actions)
        estimates = self.action_value_function(states).gather(1,actions)
        targets = torch.cat(targets)
        criterion = torch.nn.MSELoss()
        loss = criterion(estimates, targets.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.rewards_actual = []
        self.rewards_estimated = []
        self.dfs = []
        
        return loss.item()
    
def trainAgent(agent, num_episodes, env):
    rewards_over_episodes = []
    losses_over_updates = []
    for i in range(int(num_episodes)):
        obs, info = env.reset()
        rewards_in_episode = []
        rewards_actual = []
        actions_taken = []
        states_seen = []
        dfs = []
        done = False
        df = 1.0
        while not done:
            action = agent.getAction(env, obs)
            actions_taken.append(action)
            states_seen.append(torch.tensor(obs, device=agent.device).unsqueeze(0))
            obs, reward, terminated, truncated, info = env.step(action.item())
            rewards_actual.append(torch.tensor([df*(reward+truncated*(-100))], device=agent.device))
            dfs.append(torch.tensor([df], device=agent.device))
            df *= agent.gamma
            rewards_in_episode.append(reward)
            done = terminated or truncated
        dfs = torch.cat(dfs)
        rewards_actual = torch.cumsum(torch.flip(torch.cat(rewards_actual),[0]),0)
        rewards_actual = torch.flip(rewards_actual,[0])/dfs
        rewards_actual = rewards_actual.float()
        rewards_actual = [t.squeeze(0) for t in torch.split(rewards_actual.unsqueeze(0),1,dim=1)]
        agent.replay_memory.extend(zip(states_seen, actions_taken, rewards_actual))
        if len(agent.replay_memory)>=100:
            for _ in range(4):
                losses_over_updates.append(agent.update())
        agent.decayEpsilon()
        rewards_over_episodes.append(rewards_in_episode)
        if i and i%100 == 0:
            print("Average loss in last 100 updates is: {}".format(np.mean(losses_over_updates[-100:])))
            rewards = 0
            for Rs in rewards_over_episodes[-100:]:
                rewards += np.sum(Rs)
            rewards /= 100
            print("Average Total reward for last 100 episodes is: {}".format(rewards))
    return rewards_over_episodes


def main():
    env = gym.make("LunarLander-v2", continuous=False)
    agent = RLAgent()
    num_episodes = 75000
    rewards_data = trainAgent(agent, num_episodes, env)
    env.close()
    torch.save(agent.action_value_function.state_dict(),"MonteCarlo_QFunc3.pth")
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
    plt.savefig("rewards_MonteCarlo_Qlearning3.png")
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
    testModel("offPolicy_Sarsa0_QFunc.pth")
    # main()