import numpy as np
import matplotlib.pyplot as plt
import random 
array_A = np.random.rand(125)
array_B =np.random.rand(125)
plt.plot(array_A,array_B,color="red")
plt.show()
plt.figure(figsize=(7,6))
plt.bar(array_A,array_B,color="blue",edgecolor="black")
plt.show()
plt.scatter(array_A,array_B)
plt.show()
