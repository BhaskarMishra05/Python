mytuple=("A","B","C","D","E","O","P","W","Q","7","G",9,3,4,5,2,3,5,7,"A","A","A","A","F")
print(mytuple)
print("Length of tuple: ", len(mytuple))
print("Type of tuple: ",type(mytuple))
print("Access tuple item: ",mytuple[2])
print("Access tuple item from a range : ",mytuple[2:8])
print()
# Change tuple values
print("Changing values: ")
print("""A tuple cannot be changed on it's own , 
      first convert it to a list and then make changes on the 
      list ,then convert that list back to tuple""")
mylist_var=list(mytuple)
mylist_var[3]=456
mylist_var.append("XUC") #Changes at the end
mytuple=tuple(mylist_var)
print("Changed tuple: ",mytuple)
print()
#Adding two tuple
tuple1=(1,2,3)
tuple2=(3,5,0,7,6)
tuple3=tuple1+tuple2
tuple4=tuple3*2
print("Joining tuple1 and tuple2 into tuple3: ",tuple3)
print()
print("Multipling tuple: ",tuple4)
print()

#Loop through tuple
print("Loop starts from here")
for x in mytuple:
    print(x)
    
print("Loop ends here: ")
print()

print("Loop through index value: ")    
print("Loop starts from here")
for x in range(len(mytuple)):
    print(mytuple[x])
    
print("Loop ends here: ")
print()
#Count
print("Counting the number of times a value occures in tuple",mytuple.count("A"))
print()
