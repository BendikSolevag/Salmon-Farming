import torch
import numpy as np

from config import (
  TIMESTEPS_PER_ANNUM,
  COST_SMOLT,
  COST_FEED,
  COST_FIXED_HARVEST,
  N_TANKS,
  MAX_BIOMASS_PER_TANK,
  MAX_BIOMASS_FACILITY,

  SPOT_THETA,
  SPOT_SIGMA, 
  SPOT_DT, 
  
)


class Facility:
  def __init__(self):
    self.COST_SMOLT = torch.tensor(COST_SMOLT)
    self.COST_FEED = torch.tensor(COST_FEED)
    self.COST_FIXED_HARVEST = torch.tensor(COST_FIXED_HARVEST)
    self.N_TANKS = torch.tensor(N_TANKS)
    self.MAX_BIOMASS_PER_TANK = torch.tensor(MAX_BIOMASS_PER_TANK)
    self.MAX_BIOMASS_FACILITY = torch.tensor(MAX_BIOMASS_FACILITY)

    # Each individual tank is given as a list of floating point numbers. 
    # The facility state tank_fish becomes a list of lists of numbers.
    self.tank_fish = [[] for _ in range(self.N_TANKS)]
    self.growth_table = [
      [30, 1.0257],
      [100, 1.0231],
      [200, 1.0207],
      [300, 1.0188],
      [400, 1.0172],
      [500, 1.0159],
      [600, 1.0148],
      [700, 1.0139],
      [800, 1.0131],
      [900, 1.0124],
      [1000, 1.0118],
      [1100, 1.0112],
      [1200, 1.0107],
      [1300, 1.0103],
      [1400, 1.0099],
      [1500, 1.0095],
      [1600, 1.0092],
      [1700, 1.0089],
      [1800, 1.0086],
      [1900, 1.0084],
      [2000, 1.0081],
      [2250, 1.0076],
      [2500, 1.0072],
      [2750, 1.0068],
      [3000, 1.0064],
      [3250, 1.0062],
      [3500, 1.0059],
      [3750, 1.0057],
      [4000, 1.0055],
      [4250, 1.0053],
      [4500, 1.0052],
      [4750, 1.0050],
      [5000, 1.0049],
      [5250, 1.0048]
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
        weight, penalties = self.harvest(tank_population, tank_control[i], i*1000, (i+1)*1000)
        harvest_weight += weight
        missing_fish_penalties += penalties
        
      # Add smolt to tank (We do this last to avoid iterating over the smolt unneccesarily)
      tank_population += [30.0 for _ in range(tank_control[0])]
    
    return harvest_weight, missing_fish_penalties


  def grow(self):
    """
      Iterates through each tank's populations and applies the growth table (skretting) to the fish.
    """
    for tank_i in range(len(self.tank_fish)):
      for fish_i in range(len(self.tank_fish[tank_i])):
          current = self.tank_fish[tank_i][fish_i]
          rate = 1.0257
          j = 0
          # Iterate through the growth table, comparing the current fish's size to growth table weight groups. 
          while j < len(self.growth_table) - 1 and current > self.growth_table[j][0]:
            j += 1
          rate = self.growth_table[j][1]
          # Apply the identified growth rate to the fish
          self.tank_fish[tank_i][fish_i] *= rate
  
  
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
      

    # TODO: Consider using shape (2*N_TANKS, 8), rather than (N_TANKS, 16)
    return out
        



def oup():
  i = 0
  x = 0
  f = np.sin(0)
  spot = x + f

  while True:
    #print('x', x)
    #print('f', f)
    #print('spot', spot)
    yield spot
    i = i + 1
    dx = -SPOT_THETA * x * SPOT_DT + SPOT_SIGMA * np.sqrt(SPOT_DT) * np.random.normal()
    x = x + dx
    f = np.sin(((2*3.14)/TIMESTEPS_PER_ANNUM)*i) + (0.02 / TIMESTEPS_PER_ANNUM) * i
    spot = x + f



