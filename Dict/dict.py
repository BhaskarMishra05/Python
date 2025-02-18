mydict={"brand" : "Ford" , "year" : 1987,"Model" : "Xshjo"}
print(mydict)
print("Length of dict: " , len(mydict))
print("Type of dict: ", type(mydict))
print()
#Changing values
print("Updating the value of year from '1987' to '2000' ")

mydict["year"]=2000
mydict.update({"Model" : "FGHIO"})
print()

#Removing items
# mydict.pop()
# del dict - delets the whole dict.
# mydict.clear - empties the whole dict.
# del mydict["year"] - removes only year , instead of deleting whole dict