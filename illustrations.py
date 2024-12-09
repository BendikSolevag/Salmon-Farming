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
plt.savefig('./figures/growth.png')