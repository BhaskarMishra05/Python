# Basic
stringset="Hello this is a string"
print(stringset)
print("Length of string: " ,len(stringset))
print("Type of string: " , type(stringset))
print()

#Loop through string
print("Starting of loop: ")
for x in stringset:
    print(x)
print("Ending of loop: ")    
print()


# Check in or not in
txt = "Why are you running?"
print("running" in txt) # returns true if true
print("gae" not in txt) # returns true if true
print()

#Slicing
print( "2 to 5 indexing : ",txt[2:5])
print("Beginning to 7th index : " ,txt[:7])
print("Negative indexing: ", txt[-2:-9])
print()

#Upper case Lower case
print("CAPITAL LETTERS: " ,txt.upper())
print("small letters: " , txt.lower())
print()

#Replacing a string value
print(txt.replace("running" , "bunning"))
print()

#Split string i.e making a substring out of  a string
print("Spliting string into substrings: ",txt.split(","))

#With this all the methods and ways of working and modifing a strings are complete.