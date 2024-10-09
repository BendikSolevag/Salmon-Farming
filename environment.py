import torch
import numpy as np

from config import (
  MISSING_FISH_PENALTY_FACTOR,
  TIMESTEPS_PER_ANNUM,
  COST_SMOLT,
  COST_FEED,
  COST_FIXED_HARVEST,
  N_TANKS,
  MAX_BIOMASS_PER_TANK,
  MAX_BIOMASS_FACILITY,

  SPOT_KAPPA,
  SPOT_SIGMA, 
  SPOT_DT, 
  F_PARAMS, 
  
)


class Facility:
  def __init__(self):
    self.COST_SMOLT = torch.tensor(COST_SMOLT)
    self.COST_FEED = torch.tensor(COST_FEED)
    self.COST_FIXED_HARVEST = torch.tensor(COST_FIXED_HARVEST)
    self.N_TANKS = torch.tensor(N_TANKS)
    self.MAX_BIOMASS_PER_TANK = torch.tensor(MAX_BIOMASS_PER_TANK)
    self.MAX_BIOMASS_FACILITY = torch.tensor(MAX_BIOMASS_FACILITY)
    self.spot = oup()
    self.price = 0

    # Each individual tank is given as a list of floating point numbers. 
    # The facility state tank_fish becomes a list of lists of numbers.
    self.tank_fish = [[] for _ in range(self.N_TANKS)]
    
    # Monthly
    self.growth_table = [
      [30, 2.1409704122894846],
      [100, 1.9840039843125457],
      [200, 1.84902780993551],
      [300, 1.7485098723624994],
      [400, 1.6679788716416568],
      [500, 1.605198738393945],
      [600, 1.5538666885933925],
      [700, 1.5130514345310946],
      [800, 1.4776427954943907],
      [900, 1.4473184372772525],
      [1000, 1.421805703928561],
      [1100, 1.3967279635884098],
      [1200, 1.3761569602494426],
      [1300, 1.3599113224571873],
      [1400, 1.3438511460111304],
      [1500, 1.3279743858738762],
      [1600, 1.3161859647801994],
      [1700, 1.30449873120641],
      [1800, 1.2929118462974065],
      [1900, 1.285242594404863],
      [2000, 1.2738211113433746],
      [2250, 1.255003002347366],
      [2500, 1.2401422768623613],
      [2750, 1.2254517216244096],
      [3000, 1.2109294544629676],
      [3250, 1.2037308462295266],
      [3500, 1.193010440738948],
      [3750, 1.1859148405169895],
      [4000, 1.1788600437492234],
      [4250, 1.1718458238403355],
      [4500, 1.1683538597289669],
      [4750, 1.1614000828953421],
      [5000, 1.1579382142799672],
      [5250, 1.1544863217283907]
    ]

  def harvest(self, population, to_harvest, lo, hi):
      """
      Iterates over fish population in a single tank. 
      Harvests given number of fish from given weight class. 
      Returns weight of fish harvested, and potential missing fish.
      """
      harvest_weight = 0
      identified = 0
      fish_i = 0
      # Iterate through the population until we have harvested the specified number of fish
      while identified < to_harvest and fish_i < len(population):
        fish = population[fish_i]
        if fish >= lo and fish < hi:
          # If the evaluated fish fits the size requirements, add its weight to the counter and remove the fish from the population.
          identified += 1
          harvest_weight += fish
          population.pop(fish_i)
          continue
        fish_i += 1
      
      # If the control matrix specifies to harvest more fish than are available in the population, this should penalize the reward function.
      penalties = to_harvest - identified
      return harvest_weight, penalties
  

  def control(self, control_matrix):
    """
    @param control_matrix: Two dimensional array. 
      First dimension denotes tanks. 
      Second dimension controls. 
        First position determines how many smolt to release into the tank. 
        Other positions determines how many fish to harvest from each weight class.
    """
    harvest_weight = 0
    missing_fish_penalties = 0

    for tank_i in range(len(control_matrix)):
      tank_control = control_matrix[tank_i]

      # Iterate over weight classes. 1kg+, 2kg+, 3kg+, 4kg+, 5kg+, 6kg+
      tank_population = self.tank_fish[tank_i]
      for i in range(1, 7):
        # Identify the weight harvested for the given weight class, and number of penalties
        to_harvest = int(tank_control[i])
        weight, penalties = self.harvest(tank_population, to_harvest, i*1000, (i+1)*1000)
        harvest_weight += weight
        missing_fish_penalties += penalties
        
      # Add smolt to tank (We do this last to avoid iterating over the smolt unneccesarily)
      to_release = int(tank_control[0])
      if (to_release > 0):
        tank_population += [30.0 for _ in range(to_release)]
    
    return harvest_weight, missing_fish_penalties



  def grow(self):
    """
      Iterates through each tank's populations and applies the growth table (skretting) to the fish.
    """
    lastbase = self.growth_table[-1][1] - 1
    for tank_i in range(len(self.tank_fish)):
      for fish_i in range(len(self.tank_fish[tank_i])):
          current = self.tank_fish[tank_i][fish_i]
          if current > 5500:
            rate = 1 + (lastbase * ( (2 * (7500 - current)) / 7500)**2)
            self.tank_fish[tank_i][fish_i] *= rate
            continue

          rate = 1.0257
          j = 0
          # Iterate through the growth table, comparing the current fish's size to growth table weight groups. 
          while j < len(self.growth_table) - 1 and current > self.growth_table[j][0]:
            j += 1
          rate = self.growth_table[j][1]
          # Apply the identified growth rate to the fish
          self.tank_fish[tank_i][fish_i] *= rate
    self.price = next(self.spot)
  
  
  def model_input(self):
    """
    @returns Tensor with size (N_TANKS, 16), each row i denoting an individual tank, 
      each column j denoting the number of fish in each weight class, and the average weight of each fish
    """
    out = torch.zeros((self.N_TANKS, 16))

    for i in range(len(self.tank_fish)):
      tank = self.tank_fish[i]
      for fish in tank:

        # Increment the weight group
        out[i, 2 * int(fish // 1000)] += 1

        # Increment the weight count
        out[i, 2 * int(fish // 1000) + 1] += fish
      
      # Use average weight rather than total weight
      for j in range(8):
        if out[i, 2*j] != 0.0:
          out[i, (2*j)+1] = out[i, (2*j)+1] / out[i, 2*j]
      

    # TODO: Flatten state variables to make more compatible with linear layers passing
    return out
  

  def harvest_yield(self, harvest_weight):
    return harvest_weight * self.price

  
  def reward(self, harvest_weight, penalties):

    # Positive reward for selling fish
    revenue = self.price * harvest_weight

    # Penalise reward when attempting to sell fish which does not exist
    missing_fish_penalty = penalties * MISSING_FISH_PENALTY_FACTOR

    # Penalise reward constantly to avoid network doing nothing (These are the fixed running costs. Wages, electricity, etc.)
    do_nothing_bias = 1


    return revenue - missing_fish_penalty - do_nothing_bias
    

def f(t, a0, b0, a1, theta):
  omega = 2*np.pi/TIMESTEPS_PER_ANNUM,
  return a0 + b0*t + a1*np.sin(omega*t + theta)

def oup():
  i = 0
  x = 0
  spot = x + f(i, *F_PARAMS)

  while True:
    #print('x', x)
    #print('f', f)
    #print('spot', spot)
    yield spot
    i = i + 1
    dx = -SPOT_KAPPA * x * SPOT_DT + SPOT_SIGMA * np.sqrt(SPOT_DT) * np.random.normal()
    x = x + dx
    spot = x + f(i, *F_PARAMS)

