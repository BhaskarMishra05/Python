def global_var():
    global x
    x = 10
    print("Inside function " , x)
global_var()    
print("Outside function  " ,x)