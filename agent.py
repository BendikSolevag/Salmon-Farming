import config
import torch
from torch import nn

class PolicyNetwork(nn.Module):

  def __init__(self, shape):
    super(PolicyNetwork, self).__init__()
    self.activation = nn.ReLU()

    self.shape = config.N_TANKS * 16
    self.n_layers = config.POLICY_NETWORK_LAYERS
    self.layers = nn.ModuleList([
      nn.Linear(shape, shape) for _ in range(config.POLICY_NETWORK_LAYERS)
    ])
    self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, state):
    x = state
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i < self.n_layers - 1:
        x = self.activation(x)
    return x
  

class ValueNetwork(nn.Module):
  def __init__(self, shape):
    super(ValueNetwork, self).__init__()
    self.activation = nn.ReLU()

    self.shape = config.N_TANKS * 16
    self.n_layers = config.POLICY_NETWORK_LAYERS
    self.layers = nn.ModuleList([
      nn.Linear(shape, shape) for _ in range(config.POLICY_NETWORK_LAYERS - 1)
    ])
    self.layers.append(nn.Linear(shape, 1))

    self.optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, state):
    x = state
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i < self.n_layers - 1:
        x = self.activation(x)
    return x



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






