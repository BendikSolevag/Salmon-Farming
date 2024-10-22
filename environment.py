import torch
import numpy as np
import threading
import copy

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
    self.COST_FIXED_HARVEST = COST_FIXED_HARVEST.clone().detach()
    self.N_TANKS = torch.tensor(N_TANKS)

    self.MAX_BIOMASS_PER_TANK = torch.tensor([MAX_BIOMASS_PER_TANK for _ in range(N_TANKS)])

    self.MAX_BIOMASS_FACILITY = torch.tensor(MAX_BIOMASS_FACILITY)
    self.spot = oup()
    self.price = next(self.spot)

    self.plant_penalty_matrix = torch.zeros((N_TANKS, 8))
    self.plant_penalty_matrix[:, 0] = COST_SMOLT

    # Each individual tank is given as a list of floating point numbers. 
    # The facility state tank_fish becomes a list of lists of numbers.
    self.tank_fish = [[] for _ in range(self.N_TANKS)]
    self.plants = []
    self.harvests = []
    
    # Weekly
    self.growth_table = [
      (0.03, 1.1943799068682943),
      (0.1, 1.1733473387664075),
      (0.2, 1.1542152374986576),
      (0.3, 1.1392592253424174),
      (0.4, 1.1267938307200838),
      (0.5, 1.1167519571702413),
      (0.6, 1.1083149969537),
      (0.7, 1.1014527241658891),
      (0.8, 1.095383532071859),
      (0.9, 1.0900965254967199),
      (1.0, 1.0855822295154363),
      (1.1, 1.081083966926533),
      (1.2, 1.0773476282394827),
      (1.3, 1.0743665318159294),
      (1.4, 1.0713925086772824),
      (1.5, 1.0684255448322806),
      (1.6, 1.066204946205858),
      (1.7, 1.0639883046889786),
      (1.8, 1.0617756144035877),
      (1.9, 1.060302679775683),
      (2.0, 1.0580965618327276),
      (2.25, 1.0544284414614271),
      (2.5, 1.0515017981458057),
      (2.75, 1.048582120260834),
      (3.0, 1.045669393986223),
      (3.25, 1.0442156333899617),
      (3.5, 1.0420382408261932),
      (3.75, 1.0405888088275994),
      (4.0, 1.0391411052580715),
      (4.25, 1.0376951283996605),
      (4.5, 1.036972786950638),
      (4.75, 1.0355293969407338),
      (5.0, 1.0348083479512196),
      (5.25, 1.03408772935305)
      ]


  def harvest(self, population, to_harvest):
      """
      Iterates over fish population in a single tank. 
      Returns weight of fish harvested.
      """
      if to_harvest > len(population):
        harvestables = population
        population = []
        return harvestables, population

      harvestables = population[:to_harvest]
      population = population[to_harvest:]
      return harvestables, population
  


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
      to_plant, to_harvest = int(tank_control[0]), int(tank_control[1])

      self.plants.append(to_plant)
      self.harvests.append(to_harvest)

      
      altered_population = copy.deepcopy(tank_population)
      harvestables = []
      if to_harvest > 0:
        harvestables, altered_population = self.harvest(tank_population, to_harvest)    
      harvestables_global.append(harvestables)   

      
      # Add smolt to tank (We do this last to avoid iterating over the smolt unneccesarily)  
      if (to_plant > 0):
        altered_population = altered_population + [0.03 for _ in range(to_plant)]

      self.tank_fish[tank_i] = altered_population
      
    maxlength = len(max(harvestables_global, key=len))
    usable = torch.zeros((len(control_matrix), max(maxlength, 1)))
    if maxlength > 0:
      for i in range(len(control_matrix)):
        for j in range(len(harvestables_global[i])):
          usable[i, j] = harvestables_global[i][j]

    # Calculate revenue from selling fish at current spot price
    mean_tank = torch.mean(usable, 1)
    #print('mean tank', mean_tank, 'control matrix', control_matrix)
    revenue_per_tank = mean_tank * control_matrix[:, 1]
    #print('revenue per tank', revenue_per_tank)
    revenue = self.price * torch.sum(revenue_per_tank) 


    # Calculate penalty from based on biomass constraints
    resulting_tank_weights = torch.tensor([sum(tank) for tank in self.tank_fish])
    resulting_total_weight = torch.sum(resulting_tank_weights)
    per_tank_penalty = torch.sum(torch.clamp((resulting_tank_weights - self.MAX_BIOMASS_PER_TANK), 0, None))
    total_penalty = torch.clamp(resulting_total_weight - self.MAX_BIOMASS_FACILITY, 0, None)
    

    # Penalise cost of planting
    plant_matrix = control_matrix[:, 0]
    plant_matrix = torch.where(plant_matrix > 0, plant_matrix, 0.)
    plant_penalty = torch.sum(plant_matrix) * torch.tensor(COST_SMOLT)


    # Fixed cost due to harvesting
    harvest_penalty = control_matrix[:, 1]
    harvest_penalty = torch.sum(torch.where(control_matrix > 1, 1, 0.))
    harvest_penalty = harvest_penalty * self.COST_FIXED_HARVEST

    

    # Penalise doing nothing to enoucrage planting
    do_nothing_bias = torch.tensor(1)

    
    #print('revenue', revenue)
    #print('plant penalty', plant_penalty)
    #print('tank volume penalty', per_tank_penalty)
    #print('total volume penalty', total_penalty)

    
    reward = \
      revenue \
      - harvest_penalty \
      - per_tank_penalty \
      - total_penalty \
      - do_nothing_bias
    return reward
  
  
  def model_input(self):
    """
    @returns Tensor with size (N_TANKS, 16), each row i denoting an individual tank, 
      each column j denoting the number of fish in each weight class, and the average weight of each fish
    """
    #TODO: Legg til spotpris i model_input
    #TODO: vurder: Legg til std i model_input
    out = torch.zeros((self.N_TANKS, 17))
    out[:, -1] = self.price
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


  def grow(self):
    """
      Iterates through each tank's populations and applies the growth table (skretting) to the fish.
    """
    threads = []
    for tank_i in range(len(self.tank_fish)):
      t = threading.Thread(target=grow_tank, args=(self.tank_fish[tank_i], self.growth_table))
      t.start()
      threads.append(t)
    for thread in threads:
      thread.join()
      
    self.price = next(self.spot)


def grow_tank(tank: list[float], growth_table: list[(int, float)]):
  lastbase = growth_table[-1][1] - 1
  for fish_i in range(len(tank)):
    current = tank[fish_i]
    if current > 5.500:
      rate = 1 + (lastbase * ( (2 * (7.500 - current)) / 7.500)**2)
      tank[fish_i] *= rate
      continue

    rate = 1.0257
    j = 0
    # Iterate through the growth table, comparing the current fish's size to growth table weight groups. 
    while j < len(growth_table) - 1 and current > growth_table[j][0]:
      j += 1
    rate = growth_table[j][1]
    # Apply the identified growth rate to the fish
    tank[fish_i] *= rate




    

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

