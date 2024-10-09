from tqdm import tqdm
import config
import torch

from environment import Facility



if __name__ == '__main__':
  R_bar = 0
  env = Facility()

  for _ in tqdm(range(52 * config.EPOCHS)):
    state = env.model_input()
    action = policy.forward(state)

    weight, penalties = env.control(action)

    # TODO: environment needs to control spot development. It is part of state.
    reward = env.reward(weight, penalties)

    updated_state = env.model_input()
    delta = reward - R_bar + value.forward(updated_state) - value.forward(state)

    R_bar += alpha**(R_BAR) * delta

    policy.backward(state, action, delta)
    policy.backward(state, action, delta)



    state = updated_state



