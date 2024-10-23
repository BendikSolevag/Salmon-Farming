from environment import Facility
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
env = Facility()

action = torch.tensor([[1, 0]])
env.control(action)

weights = []
weights.append(env.tank_fish[0][0])

for _ in tqdm(range(200)):
  env.grow()
  weights.append(env.tank_fish[0][0])  

plt.plot(weights)
plt.show()

