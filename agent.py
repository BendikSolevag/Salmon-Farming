import torch

class MemoryBuffer:
  def __init__(self, state_action_shape, max_size):
    self.max_size = max_size,
    self.index = 0
    self.state_memory = torch.tensor((self.max_size, *state_action_shape))
    self.action_memory = torch.tensor((self.max_size, *state_action_shape))
    self.reward_memory = torch.tensor((self.max_size))
    self.next_state_memory = torch.tensor((self.max_size, *state_action_shape))

  def store(self, state, action, reward, next_state):
    index = self.index % self.max_size

    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.next_state_memory[index] = next_state

    self.index += 1
  
  def sample(self, batch_size):
    max_mem = min(self.index, self.max_size)
    batch = torch.randint(max_mem, (batch_size,))

    states = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    next_states = self.next_state_memory[batch]

    return states, actions, rewards, next_states






