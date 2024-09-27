from config import TIMESTEPS_PER_ANNUM
from environment import Facility, oup
from tqdm import tqdm
import matplotlib.pyplot as plt




fac = Facility()


fac.control([[8, 0, 0, 0, 0, 0, 0, 0]])

for _ in range(200):
  fac.grow()

out = fac.model_input()
print(out)



