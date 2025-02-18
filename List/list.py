# Let's work on List now
mylist=["Apple","Banana","Cherry","Watermelon"]
print(mylist)
print("Access items based on index value: ",mylist[0])
print()
print("Length of list: ",len(mylist))
print("Type of list: ", type(mylist))
print()

# Replacing item 
mylist[0] = "Melon" #Replacing value of "Apple " with "Melon"
print("Melon instead of Apple" , mylist)
print()
mylist.insert(1,"Apple") #Adds an item to a specified index value.
print(mylist)
print()
mylist.append("Mango")
print("Append adds at the end of list: ",mylist)
print()
mylist.remove("Banana")
print("Remove the first occurence of BANANA: ",mylist)
print()
mylist.pop()
print("POP() without index value removes the item from last: ", mylist)

print()
print("LOOP THROUGH")
print("Loop starts from here: ")
for x in mylist:
    print(x)
    
print("Loop ends here ") 

print()
#Looping through index values
print("Loop through index value: ")
print("Loop starts from here: ")
for x in range(len(mylist)):
    print(mylist[x])
print("Loop ends here")
print()
#Sorting a list
print("Sorting list: ")
txt=["A","B","C","D","E","H","Z","I","P","U","J","A"]
txt.sort()
print("Sorted list in ascending order: ",txt)
txt.sort(reverse=True)
print("Sorted list in descending order: ",txt)
txt.reverse()
print("Sorted list in reverse order: ",txt)
print()
# Copy a list
copylist=mylist.copy()
print("Copy of list: ",copylist)
copylist2=list(mylist)
print("Copy of list with the help of list() constructor: " , copylist2)
print()



