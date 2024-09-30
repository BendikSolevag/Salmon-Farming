from config import TIMESTEPS_PER_ANNUM
from environment import Facility, oup
from tqdm import tqdm
import matplotlib.pyplot as plt




fac = Facility()



fac.control([[1, 0, 0, 0, 0, 0, 0, 0]])

xs = []
xs.append(fac.tank_fish[0][0])
for _ in range(100):
  fac.grow()
  xs.append(fac.tank_fish[0][0])

print(fac.tank_fish)

plt.plot(xs)
plt.show()

