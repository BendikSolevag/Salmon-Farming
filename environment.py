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

    self.MAX_BIOMASS_PER_TANK = torch.tensor([MAX_BIOMASS_PER_TANK for _ in range(N_TANKS)])

    self.MAX_BIOMASS_FACILITY = torch.tensor(MAX_BIOMASS_FACILITY)
    self.spot = oup()
    self.price = 0

    self.plant_penalty_matrix = torch.zeros((N_TANKS, 8))
    self.plant_penalty_matrix[:, 0] = COST_SMOLT

    # Each individual tank is given as a list of floating point numbers. 
    # The facility state tank_fish becomes a list of lists of numbers.
    self.tank_fish = [[] for _ in range(self.N_TANKS)]
    
    # Weekly
    self.growth_table = [
      [30, 1.1943799068682943]
      [100, 1.1733473387664075]
      [200, 1.1542152374986576]
      [300, 1.1392592253424174]
      [400, 1.1267938307200838]
      [500, 1.1167519571702413]
      [600, 1.1083149969537]
      [700, 1.1014527241658891]
      [800, 1.095383532071859]
      [900, 1.0900965254967199]
      [1000, 1.0855822295154363]
      [1100, 1.081083966926533]
      [1200, 1.0773476282394827]
      [1300, 1.0743665318159294]
      [1400, 1.0713925086772824]
      [1500, 1.0684255448322806]
      [1600, 1.066204946205858]
      [1700, 1.0639883046889786]
      [1800, 1.0617756144035877]
      [1900, 1.060302679775683]
      [2000, 1.0580965618327276]
      [2250, 1.0544284414614271]
      [2500, 1.0515017981458057]
      [2750, 1.048582120260834]
      [3000, 1.045669393986223]
      [3250, 1.0442156333899617]
      [3500, 1.0420382408261932]
      [3750, 1.0405888088275994]
      [4000, 1.0391411052580715]
      [4250, 1.0376951283996605]
      [4500, 1.036972786950638]
      [4750, 1.0355293969407338]
      [5000, 1.0348083479512196]
      [5250, 1.03408772935305]
      ]


  def harvest(self, population, to_harvest):
      """
      Iterates over fish population in a single tank. 
      Returns weight of fish harvested.
      """
      if to_harvest > len(population):
        harvestables = population
        population = []
        return harvestables

      harvestables = population[:to_harvest]
      population = population[to_harvest:]
      return harvestables
        

  def control(self, control_matrix):
    """
    @param control_matrix: Two dimensional array. 
      First dimension denotes tanks. 
      Second dimension controls. 
        First position determines how many smolt to release into the tank. 
        Other positions determines how many fish to harvest from each weight class.
    """

    harvestables_global = []
    
    for tank_i in range(len(control_matrix)):
      tank_control = control_matrix[tank_i]

      # Iterate over weight classes. 1kg+, 2kg+, 3kg+, 4kg+, 5kg+, 6kg+
      tank_population = self.tank_fish[tank_i]
      to_harvest = int(tank_control[1])
      
      harvestables = self.harvest(tank_population, to_harvest)
      harvestables_global.append(harvestables)
        
      # Add smolt to tank (We do this last to avoid iterating over the smolt unneccesarily)
      to_release = int(tank_control[0])
      if (to_release > 0):
        tank_population += [30.0 for _ in range(to_release)]

    maxlength = len(max(harvestables_global, key=len))
    usable = torch.zeros((len(control_matrix), max(maxlength, 1)))
    if maxlength > 0:
      for i in range(len(control_matrix)):
        for j in range(len(harvestables_global[i])):
          usable[i, j] = harvestables_global[i][j]

    # Calculate revenue from selling fish at current spot price
    mean_tank = torch.mean(usable, 1)      
    revenue_per_tank = mean_tank * control_matrix[:, 1]
    revenue = self.price * torch.sum(revenue_per_tank) 


    # Calculate penalty from based on biomass constraints
    resulting_tank_weights = torch.tensor([sum(tank) for tank in self.tank_fish])
    resulting_total_weight = torch.sum(resulting_tank_weights)
    per_tank_penalty = torch.sum(torch.clamp((resulting_tank_weights - self.MAX_BIOMASS_PER_TANK), 0, None))
    total_penalty = torch.clamp(resulting_total_weight - self.MAX_BIOMASS_FACILITY, 0, None)

    # Penalise cost of planting
    plant_penalty = torch.sum(control_matrix[:, 0]) * torch.tensor(COST_SMOLT)

    # Penalise doing nothing to enoucrage planting
    do_nothing_bias = torch.tensor(1)

    
    
    reward = revenue \
      - plant_penalty  \
      - per_tank_penalty \
      - total_penalty \
      - do_nothing_bias
    return reward



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
    #TODO: Legg til spotpris i model_input
    #TODO: vurder: Legg til std i model_input
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

    

def f(t, a0, b0, a1, theta):
  omega = 2*np.pi/TIMESTEPS_PER_ANNUM
  return a0 + b0*t + a1*np.sin(omega*t + theta)

def oup():
  i = 0
  x = 0
  spot = x + f(i, *F_PARAMS)

  while True:
    yield spot
    i = i + 1
    dx = -SPOT_KAPPA * x * SPOT_DT + SPOT_SIGMA * np.sqrt(SPOT_DT) * np.random.normal()
    x = x + dx
    spot = x + f(i, *F_PARAMS)

