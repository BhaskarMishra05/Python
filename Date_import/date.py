import datetime
x = datetime.datetime.now()
print(x)
print("Year: ", x.year)
print("Day: " , x.strftime("%A"))
print("Month: ", x.strftime("%B"))
print("Century: ",x.strftime("%C"))
print()
y=datetime.datetime(2025,2,28)
print("To set a date: ",y)