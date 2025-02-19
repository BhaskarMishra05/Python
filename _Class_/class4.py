class person:
    def __init__(monkey,fname,lname,age):
        monkey.fname = fname
        monkey.lname= lname
        monkey.age= age
        
    def introduction(monkey):
          print(f"First name is : {monkey.fname} , last name is : {monkey.lname} , i am {monkey.age} yeras old")
          
obj=person("Monkey" , "Luffy" , 19)
print(obj.fname)
print(obj.lname)
print(obj.age)
print()
obj.introduction()
        