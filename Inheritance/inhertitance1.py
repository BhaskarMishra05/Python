class parent_class:
    def __init__(self,fname,lname):
        self.fname = fname 
        self.lname = lname
    def useless(self):
        print(self.fname , self.lname)
class child_class(parent_class):
    pass
# obj=parent_class() -> NO NEED TO CREATE AN OBJECT.
print("Make a variable and store the value of child_class in it.")
x = child_class("Bhaskar" , "Mishra")
x.useless() #Calling the required function.