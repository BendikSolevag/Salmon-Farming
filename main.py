from tqdm import tqdm
from agent import PolicyNetwork, ValueNetwork
import config
import torch
from environment import Facility


if __name__ == '__main__':
  R_bar = 0
  env = Facility()
  policy_net = PolicyNetwork()
  value_net = ValueNetwork()  

  for _ in tqdm(range(52 * config.EPOCHS)):
    state = env.model_input()
    action = policy_net.forward(state)
    print(action)

    weight, penalties = env.control(action)

    print('weight', weight, 'penalties', penalties)

    # TODO: environment needs to control spot development. It is part of state.
    reward = env.reward(weight, penalties)

    print('reward', reward)

    updated_state = env.model_input()
    delta: torch.Tensor = reward - R_bar + value_net.forward(updated_state) - value_net.forward(state)
    delta.backward()
    
    policy_net.optimizer.step()
    value_net.optimizer.step()
    policy_net.optimizer.zero_grad()
    value_net.optimizer.zero_grad()


    R_bar += config.LEARNING_RATE * delta


    state = updated_state



