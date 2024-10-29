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

  PENALTY_TANK_DENSITY,
  PENALTY_FACILITY_DENSITY
  
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
    self.tank_fish = [[0.03 for _ in range(int(MAX_BIOMASS_PER_TANK / 4.5))] for _ in range(self.N_TANKS)]

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
        harvestables = population + [0 for _ in range(to_harvest - len(population))]
        population = []
        return harvestables, population

      harvestables = population[:to_harvest]
      population = population[to_harvest:]
      return harvestables, population
  

  def control(self, control_matrix, debug=False):
    """
    @param control_matrix: Two dimensional array. 
      First dimension denotes tanks. 
      Second dimension controls. 
        First position determines how many smolt to release into the tank. 
        Other positions determines how many fish to harvest from each weight class.
    """


    
    if debug and torch.sum(control_matrix[0]) >= 1.0:
      print(np.mean(self.tank_fish[0]))

    tot_weights_tensor = torch.zeros((N_TANKS))
    
    for n in range(self.N_TANKS):
      tot_weights_tensor[n] = sum(self.tank_fish[n])

    harvest_reward = control_matrix * tot_weights_tensor
    harvest_reward = torch.sum(harvest_reward)
    revenue = harvest_reward * self.price

    for n in range(self.N_TANKS):
      if control_matrix[n] == 1.0:
        self.tank_fish[n] = [0.03 for _ in range(int(MAX_BIOMASS_PER_TANK / 4.5))]

    
    tank_density_penalty = torch.where(tot_weights_tensor > self.MAX_BIOMASS_PER_TANK, 1, 0)
    
    tank_density_penalty = torch.sum(tank_density_penalty) * PENALTY_TANK_DENSITY

    
    faicility_density_penalty = (torch.sum(tot_weights_tensor) > self.MAX_BIOMASS_FACILITY) * PENALTY_FACILITY_DENSITY

    # Fixed cost due to harvesting
    harvest_penalty = (torch.sum(control_matrix) > 0) * self.COST_FIXED_HARVEST
    
    reward = \
      revenue \
      - harvest_penalty \
      - tank_density_penalty \
      - faicility_density_penalty
    



    if debug and torch.sum(control_matrix) >= 1.0:
      print('reward', reward)
      print('revenue', revenue)
      print('harvest penalty', harvest_penalty)
      print('tank_density_penalty', tank_density_penalty)
      print('facility_density_penalty', faicility_density_penalty)


    
    state_ = self.model_input()
        
    return state_, reward, 0
  
  
  def model_input(self):
    """
    @returns Tensor with size (N_TANKS, 16), each row i denoting an individual tank, 
      each column j denoting the number of fish in each weight class, and the average weight of each fish
    """
    out = torch.zeros(self.N_TANKS + 1)
    out[0] = np.log(self.price)
    for i, population in enumerate(self.tank_fish):
      array = torch.tensor(population)
      mean = torch.mean(array)
      out[i+1] = mean
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

