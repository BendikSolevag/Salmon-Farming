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
  state, state_rewardable = env.model_input()
  for i in tqdm(range(52 * config.EPOCHS)):
    print('iteration', i)
    
    action = policy_net.forward(state)
    weight, penalties = env.control(action)

    # TODO: environment needs to control spot development. It is part of state.
    reward = env.reward(state_rewardable, action, penalties)

    updated_state, updated_state_rewardable = env.model_input()
    delta: torch.Tensor = reward - R_bar + value_net.forward(updated_state) - value_net.forward(state)

    critic_loss = delta**2
    actor_loss = -delta


    print(delta)
    delta.backward(retain_graph=True)

    policy_net.optimizer.step()
    value_net.optimizer.step()
    policy_net.optimizer.zero_grad()
    value_net.optimizer.zero_grad()


    R_bar += config.LEARNING_RATE * delta

    state = updated_state
    state_rewardable = updated_state_rewardable



