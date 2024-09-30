import sys
from config import MAX_BIOMASS_PER_TANK, MAX_BIOMASS_FACILITY, N_TANKS
from environment import Facility, oup
from tqdm import tqdm
import matplotlib.pyplot as plt






def traditional():

  # Find the value of a single fish when the planned farm is allowed to run for the planned 18 months.
  fac = Facility()
  fac.control([[1, 0, 0, 0, 0, 0, 0, 0]])
  for _ in range(18):
    fac.grow()
  deterministic_fish_weight = fac.tank_fish[0][0]
  print('deterministic fish weight', deterministic_fish_weight)

  fac = Facility()

  # In order to not exceed the max biomass constraints, determine the max allowable weight for each tank.
  max_w_p_t = min(MAX_BIOMASS_PER_TANK, MAX_BIOMASS_FACILITY / N_TANKS)
  print('maxwpt', max_w_p_t)

  # Floor of the  max allowable weight per tank divided by predicted fish size determines the number of fish to plant, and to harvest (as we assume 0 mortality).
  plant_n = int(max_w_p_t / deterministic_fish_weight)

  print('plant n', plant_n)

  MAX_TIMESTEPS = 90
  total_yield_per_time = [0 for _ in range(MAX_TIMESTEPS)]
  spots = [0 for _ in range(MAX_TIMESTEPS)]
  total_yield = 0

  facweight = [0 for _ in range(MAX_TIMESTEPS)]

  spot = oup()
  for t in range(MAX_TIMESTEPS):
    s = next(spot)
    spots[t] = s

    # Run fish farm for 18 month cycles
    if t == 0:
      fac.control([[plant_n, 0, 0, 0, 0, 0, 0, 0] for _ in range(N_TANKS)])
      facweight[t] = sum(fac.tank_fish[0])
      fac.grow()
      continue

    if t % 18 == 0:
      print(t)
      harvest_weight, penalties = fac.control([[plant_n, 0, 0, 0, 0, 0, plant_n, 0] for _ in range(N_TANKS)])
      harvest_yield = fac.harvest_yield(harvest_weight, s)
      total_yield += harvest_yield
      
    
    fac.grow()
    total_yield_per_time[t] = total_yield

    facweight[t] = sum(fac.tank_fish[0])


  plt.plot(total_yield_per_time)
  plt.plot(spots)
  plt.plot(facweight)
  plt.show()


