import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

N = 500
xcol = 100
def func(ls):
    ls2 = ls[:]
    return sum(ls2[50:70])+2*sum(ls2[20:30])
lsY = []
lsX = []

for i in range(N):
    lsx = []
    for j in range(xcol):
        lsx.append(random.randint(0,100))
    val = func(lsx)
    lsY.append([val])
    lsX.append(lsx)

arrY = np.array(lsY)
arrX = np.array(lsX)

df = pd.DataFrame()
df['y'] = arrY[:,0]
for i in range(xcol):
    df['x'+str(i)] = arrX[:,i]
df.to_csv('data_rand.csv',index=False)