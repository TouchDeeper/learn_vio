
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
filepath = os.path.abspath('..')+"/bin/"

mu = []
chi = []
g = []
with open(filepath + 'Radius.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        mu_ = line.split()
        numbers_float = map(float, mu_)
        # print(numbers_float)
        mu.append(numbers_float[0])
with open(filepath + 'chi.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        chi_ = line.split()
        numbers_float = map(float, chi_)
        # print(numbers_float)
        chi.append(numbers_float[0])
with open(filepath + 'g.txt', 'r') as f:
    data = f.readlines()
    for line in data:
        g_ = line.split()
        numbers_float = map(float, g_)
        # print(numbers_float)
        g.append(numbers_float[0])

x = list(range(0, len(mu), 1))
#print(x)
#print(mu)
plt.xlabel("iteration")
#plt.ylabel("radius")
plt.yscale('log')
plt.plot(x, mu, color="r", linestyle="-", marker="*", linewidth=1.0, label="radius")
plt.plot(x, chi, color="r", linestyle="-", marker="o", linewidth=1.0, label="chi")
plt.plot(x, g, color="r", linestyle="-", marker="+", linewidth=1.0, label="g")
plt.legend()
plt.show()




