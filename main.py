from environment import Facility
from tqdm import tqdm

environment = Facility()
print(environment.tank_fish)
environment.control([[10, 0, 0, 0, 0, 0, 0, 0]])

for _ in tqdm(range(400)):
  environment.grow()

print(environment.tank_fish)

weight, mistakes = environment.control([[0, 0, 0, 0, 5, 0, 0, 0]])

print(environment.tank_fish)
print(weight)
print(mistakes)