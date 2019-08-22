import matplotlib.pyplot as plt
import numpy as np
 
x = np.arange(0, 1, 0.1)
 
#y = 1/(1-pow((2*x - 1), 3))
y = 1-pow((2*x - 1), 3)
#plt.title("")
plt.plot(x, y)
plt.show()

