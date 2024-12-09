from tqdm import tqdm
from agent import PolicyNetwork, ValueNetwork
import config
import torch
from environment import Facility
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

def test():

  policy_net = PolicyNetwork()
  policy_net.load_state_dict(torch.load('./model/policy_net.pth', weights_only=True))
  
  fig, axes = plt.subplots(3, 1)

  EPOCHS = 1
  MAX_TIMESTEPS = 1000
  pbar = tqdm(total=EPOCHS * MAX_TIMESTEPS)

  env = Facility()
  env_traditional = Facility()

  state = env.model_input()
  total_tank_weights = []
  rewards = []
  accumulative_reward = 0

  trad_rewards = []
  trad_accumulative_reward = 0

  harvest_probs_history = []

  for i in range(MAX_TIMESTEPS):
    pbar.update(1)
    
    # Create 
    out = policy_net.forward(state)    
    harvest_probs_history.append(out[0].item())

    harvest_probs = torch.distributions.Bernoulli(out)
    control_matrix = harvest_probs.sample()
    
    state_, reward, done = env.control(control_matrix)    

    # Control traditional plant we're comparing ourselves to
    if i > 0 and i % 49 == 0:
      tradnewstate, tradreward, traddone = env_traditional.control(torch.ones(config.N_TANKS, dtype=torch.float))
      trad_accumulative_reward = trad_accumulative_reward + tradreward
    else:
      tradnewstate, tradreward, traddone = env_traditional.control(torch.zeros(config.N_TANKS, dtype=torch.float))
      trad_accumulative_reward = trad_accumulative_reward + tradreward
    
    # Update state variable
    env.grow()
    state = state_
    
    # Track plot data points
    total_tank_weights.append(sum(env.tank_fish) * env.PLANT_NUNMBER)
    accumulative_reward = accumulative_reward + reward.item()
    rewards.append(accumulative_reward)
    trad_rewards.append(trad_accumulative_reward)

  
  axes[0].plot(total_tank_weights, label=f"Weights")
  axes[0].plot([config.MAX_BIOMASS_FACILITY for _ in range(len(total_tank_weights))])
  #axes[0].plot(prices, label=f"Price development {epoch}")
  axes[1].plot(rewards, label=f"rewards")
  

  axes[1].plot(trad_rewards, label=f"Trad rewards")
      
  axes[2].plot(harvest_probs_history, label="Probability history")

  axes[0].legend()
  axes[1].legend()
  plt.show()




if __name__ == '__main__':
  test()