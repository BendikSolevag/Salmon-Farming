from tqdm import tqdm
from agent import PolicyNetwork, ValueNetwork
import config
import torch
from environment import Facility
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
  learning_rate = torch.tensor(config.LEARNING_RATE)
  R_bar = torch.tensor(0.0)

  env = Facility()
  policy_net = PolicyNetwork()
  value_net = ValueNetwork()  
  state = env.model_input()
  for i in tqdm(range(52 * config.EPOCHS)):
    print('iteration', i)
    
    state = env.model_input()
    out = policy_net.forward(state)[0]
    plant_mu, plant_sigma, harvest_mu, harvest_sigma = out[0], torch.exp(out[1]), out[2], torch.exp(out[3])

    plant_probs = torch.distributions.Normal(plant_mu, plant_sigma)
    plant_action = plant_probs.sample()
    plant_log_probs = plant_probs.log_prob(plant_action)
    harvest_probs = torch.distributions.Normal(harvest_mu, harvest_sigma)
    harvest_action = harvest_probs.sample()
    harvest_log_probs = harvest_probs.log_prob(harvest_action)

    
    action = torch.tensor([[plant_action, harvest_action]])


    reward = env.control(action)
    updated_state = env.model_input()



    delta = reward - R_bar + value_net(updated_state) - value_net.forward(state)

    
    R_bar = (R_bar + learning_rate * delta).detach()


    critic_loss = delta**2
    actor_loss = -(harvest_log_probs + plant_log_probs) * delta
    
    combined = actor_loss + critic_loss
    combined.backward()
    print('combined loss', combined)

    policy_net.optimizer.step()
    value_net.optimizer.step()
    policy_net.optimizer.zero_grad()
    value_net.optimizer.zero_grad()
    env.grow()
    print('learning rate', learning_rate)
    print('delta', delta.item())
