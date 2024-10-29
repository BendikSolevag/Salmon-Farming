from tqdm import tqdm
from agent import PolicyNetwork, ValueNetwork
import config
import torch
from environment import Facility
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

def train():
  learning_rate = torch.tensor(config.LEARNING_RATE)
  R_bar = torch.tensor(0.0)

  
  policy_net = PolicyNetwork()
  value_net = ValueNetwork()  
  fig, axes = plt.subplots(2, 1)
  

  EPOCHS = 9
  MAX_TIMESTEPS = 1000
  pbar = tqdm(total=EPOCHS * MAX_TIMESTEPS)
  for epoch in range(EPOCHS):
    env = Facility()
    state = env.model_input()
    total_tank_weights = []
    rewards = []
    accumulative_reward = 0
    for i in range(MAX_TIMESTEPS):
      pbar.update(1)
      
      out = policy_net.forward(state)
      harvest_probs = torch.distributions.Bernoulli(out)
      control_matrix = harvest_probs.sample()
      harvest_log_probs = harvest_probs.log_prob(control_matrix)

      
      state_, reward, done = env.control(control_matrix)    
      
      
      
      
      delta = reward - R_bar + config.DISCOUNT_RATE * value_net(state_) - value_net.forward(state)    
      
      R_bar = ((1 - learning_rate) * R_bar + learning_rate * delta).detach()

      
      critic_loss = delta**2
      
      actor_loss = -torch.sum(harvest_log_probs) * delta
      
      combined = actor_loss + critic_loss
      combined.backward()

      policy_net.optimizer.step()
      value_net.optimizer.step()
      policy_net.optimizer.zero_grad()
      value_net.optimizer.zero_grad()
      
      env.grow()
      state = state_

      
      total_tank_weights.append(sum(env.tank_fish[0]))
      accumulative_reward = accumulative_reward + reward.item()
      rewards.append(accumulative_reward)

    if epoch % 2 == 0:
      axes[0].plot(total_tank_weights, label=f"epoch {epoch}")
      axes[0].plot([config.MAX_BIOMASS_FACILITY for _ in range(len(total_tank_weights))])
      axes[1].plot(rewards, label=f"epoch {epoch}")
  

  axes[0].legend()
  axes[1].legend()
  plt.show()




if __name__ == '__main__':
  train()