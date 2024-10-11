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



    inter = reward - R_bar
    print('inter', inter)
    medi = value_net(updated_state)[0][0]
    print('medi', medi)
    ate = value_net.forward(state)[0][0]
    print('ate', ate)
    delta = inter + medi - ate
    print('delta', delta)

    critic_loss = delta**2
    actor_loss = -delta
    combined = actor_loss + critic_loss
    print(combined)
    combined.backward(retain_graph=True)

    policy_net.optimizer.step()
    value_net.optimizer.step()
    policy_net.optimizer.zero_grad()
    value_net.optimizer.zero_grad()



    print('learning rate', learning_rate)
    print('delta', delta.item())
    R_bar += learning_rate * delta

    



