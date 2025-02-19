cars=["Ford","Toyota","BMW","Maruti"]
print(cars)
x = cars[1]
print(x)
print()
print("Modifing array")
cars[0]="KIA"
print(cars)
print()
print("Length of array: ",len(cars))
print("Type of array: ", type(cars))

print()
print("Loop through Array")
print("Loop starts here: ")
for x in cars:
    print(x)
print("Loop ends here: ")
print()
print("Adding item through append")
cars.append("MG")
print()
# cars.pop(index_value)
#cars.remove("Toyota")
#cars.clear() -> Remove all elements from list
#cars.reverse() -> Reverse the order of list
#cars.sort() -> Sorts the list