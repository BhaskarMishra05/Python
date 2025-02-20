import numpy as np
# Taking input from user and making array out of it.
#first make a empty list and later convert it to array.
L=[] #empty list
for i in range(1,5): # range(1,5) means only 4 elements
    L_input=int(input("Enter: "))
    L.append(L_input) #append the value in list.
    #loop ends here
print(np.array(L)) #Converting List into an array using np.array.