import numpy as np
# With the help of .ndmin we can create any kind of dimentional array
two_d_array=np.array([1,2,3,4,5,6,7],ndmin=2)
print(two_d_array)
print()
ten_d_array=np.array([1,2,3,4,5,6,7,8,9],ndmin=10)
print(ten_d_array)
print()
thousand_d_array=np.array([1,2,3,4,5,6,7,8,9],ndmin=64) #The maximum numbers of ndmin you can create is 64.
print(thousand_d_array)
