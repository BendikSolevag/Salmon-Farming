from config import TIMESTEPS_PER_ANNUM
from environment import Facility, oup
from tqdm import tqdm
import matplotlib.pyplot as plt



xs = []
for i, spot in enumerate(oup()):
  xs.append(spot)
  print(i)
  if i > TIMESTEPS_PER_ANNUM * 5:
    break

plt.plot(xs)
plt.show()