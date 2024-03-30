import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import math

class ActionValueFunctionApproximator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n_state_space = 8
        n_action_space = 2
        hidden_space_1 = 16
        hidden_space_2 = 32

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_state_space+n_action_space, hidden_space_1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_space_1, hidden_space_2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_space_2, 1)
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
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 2500
        self.epsilon = self.epsilon_start
        self.rewards_actual = []
        self.rewards_estimated = []
        self.dfs = []


    def getEstimatedReward(self, s, a):
        state_action_pair = np.concatenate([s,a], axis=None)
        state_action_pair = torch.tensor(state_action_pair, device=self.device)
        state_action_pair = state_action_pair.float()

        return self.action_value_function(state_action_pair)


    def getAction(self, env, s):
        if np.random.random() < self.epsilon:
            action = env.action_space.sample()
            a = np.array([action&1, (action//2)&1])
            estimated_reward = self.getEstimatedReward(s,a)
            self.rewards_estimated.append(estimated_reward)
            return action
        else:
            best_a, best_reward = -1, -100000.0
            for i in range(4):
                a = np.array([i&1, (i//2)&1])
                estimated_reward = self.getEstimatedReward(s,a)
                if estimated_reward-best_reward > 1e-6:
                    best_a = i
                    best_reward = estimated_reward
            self.rewards_estimated.append(best_reward)
            return best_a

    def decayEpsilon(self, episodes_done):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end)*math.exp(-1. * episodes_done / self.epsilon_decay)

    def update(self):
        estimates = torch.cat(self.rewards_estimated)
        dfs = torch.cat(self.dfs)
        targets = torch.cumsum(torch.flip(torch.cat(self.rewards_actual),[0]),0)
        targets = torch.flip(targets,[0])/dfs
        targets = targets.float()

        criterion = torch.nn.MSELoss()
        loss = criterion(estimates, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.rewards_actual = []
        self.rewards_estimated = []
        self.dfs = []
        
        return loss
    
def trainAgent(agent, num_episodes, env):
    rewards_over_episodes = []
    for i in range(int(num_episodes)):
        obs, info = env.reset()
        rewards_in_episode = []
        done = False
        df = 1.0
        while not done:
            action = agent.getAction(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            agent.rewards_actual.append(torch.tensor([df*reward], device=agent.device))
            agent.dfs.append(torch.tensor([df], device=agent.device))
            df *= agent.gamma
            rewards_in_episode.append(reward)
            done = terminated or truncated
        loss = agent.update()
        agent.decayEpsilon(i+1)
        rewards_over_episodes.append(rewards_in_episode)
        if i and i%100 == 0:
            print("Loss in episode {} is: {}".format(i,loss))
            rewards = 0
            for Rs in rewards_over_episodes[-100:]:
                rewards += np.sum(Rs)
            rewards /= 100
            print("Average Total reward for last 100 episodes is: {}".format(rewards))



def main():
    env = gym.make("LunarLander-v2", continuous=False)
    agent = RLAgent()
    trainAgent(agent, 5e4, env)
    env.close()
    
    env = gym.make("LunarLander-v2", render_mode="human", continuous=False)
    observation, info = env.reset()
    for _ in range(1000):
        action = agent.getAction(env, observation)
        # action = heuristic(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()

if __name__ == "__main__":
    main()