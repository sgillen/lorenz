# %%

import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,10,100)
x = np.sin(t)

plt.plot(t,x)
#plt.figure()
plt.show()
