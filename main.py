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
  state, state_rewardable = env.model_input()
  for i in tqdm(range(52 * config.EPOCHS)):
    print('iteration', i)
    
    state, state_rewardable = env.model_input()
    action = policy_net.forward(state)
    weight, penalties = env.control(action)

    reward = env.reward(state_rewardable, action, penalties)
  
    updated_state, updated_state_rewardable = env.model_input()

    delta = reward - R_bar + value_net(updated_state) - value_net.forward(state)
    R_bar = (R_bar + learning_rate * delta).detach()
    

    critic_loss = delta**2
    actor_loss = -delta
    
    combined = actor_loss + critic_loss
    combined.backward()
    print('combined loss', combined)

    policy_net.optimizer.step()
    value_net.optimizer.step()
    policy_net.optimizer.zero_grad()
    value_net.optimizer.zero_grad()

    print('learning rate', learning_rate)
    print('delta', delta.item())
