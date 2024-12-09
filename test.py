from tqdm import tqdm
from agent import PolicyNetwork, ValueNetwork
import config
import torch
from environment import Facility
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
import numpy as np

def test():

  policy_net = PolicyNetwork()
  policy_net.load_state_dict(torch.load('./model/policy_net.pth', weights_only=True))
  
  fig, axes = plt.subplots(3, 1)

  EPOCHS = 1
  MAX_TIMESTEPS = 50000
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

  tank_weight_on_harvest_actions = []
  price_shock_on_harvest_actions = []
  price_deviation_from_mean_on_harvest_actions = []

  for i in range(MAX_TIMESTEPS):
    pbar.update(1)
    
    # Create 
    out = policy_net.forward(state)    
    harvest_probs_history.append(out[0].item())

    harvest_probs = torch.distributions.Bernoulli(out)
    control_matrix = harvest_probs.sample()

    if torch.sum(control_matrix) >= 1.0:
      price_shock_on_harvest_actions.append(env.price_rand)
      price_deviation_from_mean_on_harvest_actions.append(env.price - env.price_mean_year)
      for i in range(len(env.tank_fish)):
        if control_matrix[i] == 1.0:
          tank_weight_on_harvest_actions.append(env.tank_fish[i].item())
        

    
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
  #plt.show()
  plt.close()

  plt.plot(total_tank_weights)
  plt.xlabel('Time step (week)')
  plt.ylabel('Sum tank weight')
  plt.savefig('./figures/eval/tankweight.jpg', format="jpg")
  plt.close()

  plt.plot(rewards, label="Model Reward")
  plt.plot(trad_rewards, label="Traditional Reward")
  plt.xlabel('Time step (week)')
  plt.ylabel('Reward')
  plt.savefig('./figures/eval/accumulative_reward.jpg', format="jpg")
  plt.close()
  
  plt.plot(harvest_probs_history)
  plt.xlabel('Time step (week)')
  plt.ylabel('Model predicted probability')
  plt.savefig('./figures/eval/model_probability.jpg', format="jpg")
  plt.close()

  print(np.corrcoef(total_tank_weights, harvest_probs_history))

  plt.hist(tank_weight_on_harvest_actions)
  plt.xlabel('Tank weight on harvest')
  plt.ylabel('Frequency')
  plt.savefig('./figures/eval/harvest_action_histogram_weight.jpg', format="jpg")
  plt.close()

  plt.hist(price_shock_on_harvest_actions)
  plt.xlabel('Stochastic term on harvest')
  plt.ylabel('Frequency')
  plt.savefig('./figures/eval/harvest_action_histogram_price.jpg', format="jpg")
  plt.close()

  plt.hist(price_deviation_from_mean_on_harvest_actions)
  plt.xlabel('Price deviation from year mean on harvest')
  plt.ylabel('Frequency')
  plt.savefig('./figures/eval/harvest_action_histogram_price_deviation.jpg', format="jpg")
  plt.close()




if __name__ == '__main__':
  test()