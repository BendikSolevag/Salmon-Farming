from environment import Facility
import matplotlib.pyplot as plt

fac = Facility()
fac.tank_fish = [[0.03]]

weight_prog = []

for _ in range(200):
  weight_prog.append(fac.tank_fish[0][0])
  fac.grow()

plt.plot(weight_prog)
plt.xlabel('Week')
plt.ylabel('Weight')
plt.savefig('./figures/growth.png')