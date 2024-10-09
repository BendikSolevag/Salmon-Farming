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

    weight, penalties = env.control(action)

    # TODO: environment needs to control spot development. It is part of state.
    reward = env.reward(weight, penalties)

    updated_state = env.model_input()
    delta = reward - R_bar + value_net.forward(updated_state) - value_net.forward(state)

    R_bar += config.LEARNING_RATE**(R_bar) * delta


    state = updated_state



