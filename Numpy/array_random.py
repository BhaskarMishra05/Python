import numpy as np
import random
# This create  a array with random numbers.
array_one=np.random.rand(9) # 9 is the number of elements. Remember all values will be between 0 and 1.
print(array_one)
print()
array_two=np.random.rand(3,3)
print(array_two) # 3,3 are rows and columns.
print()
array_three=np.random.randint(3,20,89) #3 is starting number , 20 is ending number and 89 is number of elements.
print(array_three)
