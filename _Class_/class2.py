class person:
    def __init__(self,name,age):
        self.name = name # self can access the elements of class.
        self.age = age #Self is the first parameter of any function and we can name it anything. Lets hop to next example to see this.
        
    def intro(self):
        print(f"Hello my self {self.name} and I am {self.age} years old")    
        
obj = person("JOhn" , 38)
print(obj.name)
print(obj.age)
obj.intro() #We access the functions inside the class with the help of class object.