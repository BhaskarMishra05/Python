myset={"A","B","C","D","E","R","D",True,1,False,0}
print(myset)
print()
print("Length of Set: ",len(myset))
print("Type of Set: ",type(myset))
print()
print("z" in myset)
print("A" not in myset)
print()

#Loop
print("Loop starts here: ")
for x in myset:
    print(x)
print("Loop ends here: ")
print()
print("Looping through index is impossible in set as set is unindexiable") 
print()
# Adding items
myset.add("HOOOO") 
print(".add() adds items at the beginning")
# Removing items
myset.remove("A")
myset.discard("E")
print("A and E have been removed from the set: ", myset)
print()
print("POP() removes an item randomly: ",myset.pop())
print("Clear() empties the set : ",myset.clear())
print()
print("Additionally: ")
print()
set1=set("2,35,4,5")
print(set1)
del set1
print("With of help of del keyword we can delete a set completely")
