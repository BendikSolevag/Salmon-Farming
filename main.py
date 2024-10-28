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

  env = Facility()
  policy_net = PolicyNetwork()
  value_net = ValueNetwork()  
  state = env.model_input()
  total_tank_weights = []
  rewards = []

  MAX_TIMESTEPS = 50000

  for i in tqdm(range(MAX_TIMESTEPS)):
    
    out = policy_net.forward(state)
    harvest_probs = torch.distributions.Bernoulli(out)
    control_matrix = harvest_probs.sample()
    harvest_log_probs = harvest_probs.log_prob(control_matrix)

    reward = env.control(control_matrix)    
    updated_state = env.model_input()

    delta = reward - R_bar + value_net(updated_state) - value_net.forward(state)    
    R_bar = ((1 - learning_rate) * R_bar + learning_rate * delta).detach()
    critic_loss = delta**2
    actor_loss = -(harvest_log_probs) * delta
    combined = actor_loss + critic_loss
    combined.backward()

    policy_net.optimizer.step()
    value_net.optimizer.step()
    policy_net.optimizer.zero_grad()
    value_net.optimizer.zero_grad()
    
    env.grow()
    state = updated_state
    
    total_tank_weights.append(sum(env.tank_fish[0]))
    rewards.append(reward.item())

    
      
  fig, axes = plt.subplots(2, 1)
  
  axes[0].plot(total_tank_weights)
  axes[0].plot([config.MAX_BIOMASS_FACILITY for _ in range(len(total_tank_weights))])

  axes[1].plot(rewards)
  plt.show()




if __name__ == '__main__':
  train()