import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from environment import Facility
from tqdm import tqdm
import config

def plotLearning(scores, filename, x=None, window=5):   
  N = len(scores)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
  if x is None:
    x = [i for i in range(N)]
  plt.ylabel('Score')       
  plt.xlabel('Game')                     
  plt.plot(x, running_avg)
  plt.savefig(filename)



class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = (config.N_TANKS + 1)
        self.fc1_dims = 8
        self.fc2_dims = 8

        self.n_actions = config.N_TANKS
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, self.n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, observation):
        state = torch.tensor(observation, dtype=torch.float)
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        pi = self.sigmoid(self.pi(x))
        v = self.v(x)

        return pi, v

class Agent(object):
    def __init__(self):
        self.gamma = 0.99
        self.log_probs = None
        
        self.actor_critic = ActorCriticNetwork()

    def choose_action(self, observation):
        pi, v = self.actor_critic.forward(observation)

        action_probs = torch.distributions.Bernoulli(pi)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        
        
        reward = torch.tensor(reward, dtype=torch.float)
        
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value
        

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()


if __name__ == '__main__':
    
    agent = Agent()

    filename = 'LunarLander-ActorCriticNaiveReplay-256-256-Adam-lr00001.png'
    scores = []

    MAX_TIMESTEPS = 2500
    EPOCHS = 10
    pbar = tqdm(total = MAX_TIMESTEPS * EPOCHS)

    for i in range(EPOCHS):
      env = Facility()
      observation = env.model_input()
      total_tank_weights = []
      rewards = []
      score = 0
      
      for i in range(MAX_TIMESTEPS):
        pbar.update(1)
        print(observation)
        action = agent.choose_action(observation)
        observation_, reward, done = env.control(action)
        if i == (MAX_TIMESTEPS - 1):
           done = 1

        score += reward

        agent.learn(observation, reward, observation_, done)
        observation = observation_

      scores.append(score)
      avg_score = np.mean(scores[max(0, i-100):(i+1)])
      print('episode: ', i,'score %.1f ' % score,
            ' average score %.1f' % avg_score)

    x = [i+1 for i in range(EPOCHS)]
    plotLearning(scores, filename, x)