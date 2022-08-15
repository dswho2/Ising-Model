import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

import os
import imageio

#Ising simulation implemented with a monte carlo method, the metropolis-hastings algorithm

#bias term


epochs = 25000
T = 1 #unit: J/kb
gridSize = 64

grid = np.random.random((gridSize, gridSize))
grid[grid>0.5] = 1
grid[grid<= 0.5] = -1

def printGrid():
  for i in range(len(grid)):
    for j in range(len(grid[0])):
      print(grid[i][j], end=" | ")
    print()
  plt.imshow(grid)




def boundary(i):
  if i >= gridSize:
    return 0
  if i < 0:
    return gridSize - 1
  else:
    return i

def hamiltonian(i, j):
  # return -grid[i,j]*(grid[max(0,i - 1), j] + grid[min(gridSize-1, i + 1), j] + grid[i, max(0, j - 1)] + grid[i, min(gridSize-1, j + 1)])
  return -grid[i,j]*(grid[boundary(i - 1), j] + grid[boundary(i + 1), j] + grid[i, boundary(j - 1)] + grid[i, boundary(j + 1)]) #periodic boundary condition

filenames = []

#metropolis-hastings algorithm
def run(draw=False, frames = 500):
  j=0
  for i in range(epochs):
    x, y = np.random.randint(0, gridSize), np.random.randint(0, gridSize)

    E = -2*hamiltonian(x,y)

    if E <= 0:
      grid[x,y] *= -1
    elif np.exp(-E/T) > np.random.rand(): #monte carlo simulation
      grid[x,y] *= -1

    if i % frames == 0 and draw:
      plt.figure(figsize=(10,10))
      plt.imshow(grid)
      
      filename = f'{j}.png'
      filenames.append(filename)
      # save frame
      plt.savefig(filename)
      j+=1

      plt.show()

      time.sleep(0.01)
      clear_output(wait=True)



# Sweep Temperature
N = 11 #number of temperature points
n_avg = 5 #number of simulations to average
temperatures = np.linspace(1,10,N) #unit J/Kb
Mavg = np.zeros(N)

# initial condition
for i in range(len(temperatures)):
  T = temperatures[i]
  M = np.zeros(n_avg)
  for j in range(n_avg):
    grid = np.ones((gridSize, gridSize))
    run()
    M[j] = np.sum(grid)/(gridSize**2)
  Mavg[i] = np.mean(M)
  
plt.figure(figsize=(10,6))
plt.plot(temperatures, Mavg);
plt.xlabel('Temperature ($J/K_b$)');
plt.ylabel('$<M>$');



run(True)
# build gif
with imageio.get_writer('Ising_Simulation.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        

# Remove files
for filename in set(filenames):
    os.remove(filename)