import pandas as pd
from numba import jit, cuda 
import numpy
df=pd.read_csv('./Dataset/scaledNForkrig.csv', sep=',',header=None)
list_data = df.to_numpy().tolist()
print(list_data)
print(type(list_data))
long_data=[]
for i in range(len(list_data)):
  long_data.append([])
  for k in range(2):
    long_data[i].append(list_data[i][k])
print(long_data)
N_data=[]
for i in range(len(list_data)):
  N_data.append(list_data[i][2])
# print(N_data)


import numpy as np
import skgstat as skg

coordinates = long_data
values =N_data
@jit
def variogram(coordinates , values):
    v =  skg.Variogram(coordinates, values,normalize=False, n_lags=20, maxlag=0.5 , bin_func = 'uniform' , verbose = True)
    return v

print('sdhfurigh')
V = variogram(coordinates , values)
print(V)
V.plot()

print('dkjghiufdhgvncorguydfjklgifvhcj')