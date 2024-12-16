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
  SPOT_KAPPA,
  SPOT_SIGMA, 
  SPOT_DT, 
  F_PARAMS, 
  F_PARAMS_DETREND,
  PENALTY_TANK_DENSITY,
  PENALTY_FACILITY_DENSITY
)
#F_PARAMS = F_PARAMS_DETREND

class Facility:
  def __init__(self):
    self.COST_SMOLT = torch.tensor(COST_SMOLT)
    self.COST_FEED = torch.tensor(COST_FEED)
    self.COST_FIXED_HARVEST = COST_FIXED_HARVEST.clone().detach()
    self.N_TANKS = torch.tensor(N_TANKS)

    self.MAX_BIOMASS_PER_TANK = torch.tensor([MAX_BIOMASS_PER_TANK for _ in range(N_TANKS)])
    self.MAX_BIOMASS_FACILITY = torch.tensor(MAX_BIOMASS_FACILITY)
    
    self.spot = oup()
    self.price, self.price_rand = next(self.spot)
    self.price_mean_year = self.price
    

    self.plant_penalty_matrix = torch.zeros((N_TANKS, 8))
    self.plant_penalty_matrix[:, 0] = COST_SMOLT


    # Number of fish planted each generation
    self.PLANT_NUNMBER = int(MAX_BIOMASS_PER_TANK / 4.5)
    
    # Each individual tank is given as a floating point number, representing the weight of an individual fish in the tank. 

    self.tank_fish = torch.Tensor([0.03 for _ in range(self.N_TANKS)])
    
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


  def control(self, control_matrix, debug=False):
    """
    @param control_matrix: Two dimensional array. 
      First dimension denotes tanks. 
      Second dimension controls. 
        First position determines how many smolt to release into the tank. 
        Other positions determines how many fish to harvest from each weight class.
    """
    
    
    tot_weights_tensor = self.tank_fish * self.PLANT_NUNMBER

    # Calculate earned revenue from selling
    harvest_reward = control_matrix * tot_weights_tensor
    harvest_reward = torch.sum(harvest_reward)
    revenue = harvest_reward * self.price

    # Reset emptied tanks
    for n in range(self.N_TANKS):
      if control_matrix[n] == 1.0:
        self.tank_fish[n] = 0.03

    # Calculate density penalties where applicable
    tank_density_penalty = torch.where(tot_weights_tensor > self.MAX_BIOMASS_PER_TANK, 1, 0)
    tank_density_penalty = torch.sum(tank_density_penalty) * PENALTY_TANK_DENSITY
    facility_density_penalty = (torch.sum(tot_weights_tensor) > self.MAX_BIOMASS_FACILITY) * PENALTY_FACILITY_DENSITY

    # Add fixed cost of harvesting if applicable
    harvest_penalty = (torch.sum(control_matrix) > 0) * self.COST_FIXED_HARVEST
    

    # Sum returnable reward from components
    reward = \
      revenue \
      - harvest_penalty \
      - tank_density_penalty \
      - facility_density_penalty
    

    #debug = True
    if debug and torch.sum(control_matrix) >= 1.0:
      print('reward', reward)
      print('revenue', revenue)
      print('harvest penalty', harvest_penalty)
      print('tank_density_penalty', tank_density_penalty)
      print('facility_density_penalty', facility_density_penalty)


    # Also return next state to control system
    state_ = self.model_input()
        
    # Return (next state, reward, done)
    return state_, reward, 0
  
  
  def model_input(self):
    """
    @returns Tensor with size (N_TANKS, 16), each row i denoting an individual tank, 
      each column j denoting the number of fish in each weight class, and the average weight of each fish
    """
    out = torch.zeros(self.N_TANKS + 1)
    # Use the log-price so that the price input does not far exceed the weight input
    out[0] = np.log(self.price)
    for i, mean_weight in enumerate(self.tank_fish):
      out[i+1] = mean_weight
    return out
  


  def harvest_yield(self, harvest_weight):
    return harvest_weight * self.price


  def grow(self):
    """
      Iterates through each tank's populations and applies the growth table (skretting) to the fish.
    """
    lastbase = self.growth_table[-1][1] - 1
    for fish_i in range(len(self.tank_fish)):
      current = self.tank_fish[fish_i]
      if current > 5.500:
        rate = 1 + (lastbase * ( (2 * (7.500 - current)) / 7.500)**2)
        self.tank_fish[fish_i] *= rate
        continue

      rate = 1.0257
      j = 0
      # Identify correct growth rate to apply to the fish
      while j < len(self.growth_table) - 1 and current > self.growth_table[j][0]:
        j += 1
      rate = self.growth_table[j][1]
      self.tank_fish[fish_i] *= rate

    self.price, self.price_rand = next(self.spot)
    self.price_mean_year = (self.price_mean_year * 51) / 52 + self.price / 52





def f(t, a0, b0, a1, theta):
  omega = 2*np.pi/TIMESTEPS_PER_ANNUM
  return a0 + a1*np.sin(omega*t + theta)

def oup():
  i = 0
  x = 0
  spot = x + f(i, *F_PARAMS)
  rand = 0

  while True:
    yield spot, rand
    i = i + 1
    rand = np.random.normal()
    dx = -SPOT_KAPPA * x * SPOT_DT + SPOT_SIGMA * np.sqrt(SPOT_DT) * rand
    x = x + dx
    spot = x + f(i, *F_PARAMS)
    spot = max(1, spot)

