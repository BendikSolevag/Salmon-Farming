import numpy as np
import torch
from agent import PolicyNetwork
from environment import Facility
import matplotlib.pyplot as plt

fac = Facility()
weight_prog = []

for _ in range(125):
  weight_prog.append(fac.tank_fish[0].item())
  fac.grow()

plt.plot(weight_prog)
plt.xlabel('Week')
plt.ylabel('Weight')
plt.savefig('./figures/datasetanalysis/growth.png', format="png")
plt.close()


heatmap = np.zeros((100, 100))
policy_net = PolicyNetwork()
policy_net.load_state_dict(torch.load('./model/policy_net.pth', weights_only=True))


coarseness = 100
weight_increment = (3 - 2) / coarseness
price_increment = (120 - 60) / coarseness
heatmap = np.zeros((coarseness, coarseness))
for i in range(coarseness):
  fishweight = 2 + weight_increment * i
  for j in range(coarseness):
    price = 60 + price_increment * i
    modelinput = np.log(price)
    modelinput = torch.tensor([modelinput, fishweight], dtype=torch.float)
    out = policy_net.forward(modelinput)
    heatmap[i, j] = out[0].item()


plt.imshow( heatmap )
plt.ylabel('Weight')
plt.xlabel('Price')
plt.title( "Decision boundary" ) 
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
plt.savefig('./figures/datasetanalysis/decision_boundary.jpg', format="jpg")
plt.close()
