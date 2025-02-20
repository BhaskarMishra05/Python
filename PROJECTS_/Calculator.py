print("Basic Calculator")
print()
number_1=int(input("Enter first number: "))
number_2=int(input("Enter second number: "))
operator=input('Enter the Operator (+ ,- ,* ,/  ,% ,// ,** : ')
print()
if operator=="+":
    print(number_1+number_2)
elif operator=="-":
    print(number_1-number_2)
elif operator=="*":
    print(number_1*number_2)
elif operator=="/":
    print(number_1/number_2)
elif operator=="%":
    print(number_1%number_2)
elif operator=="//":
    print(number_1//number_2)
elif operator=="**":
    print(number_1**number_2)
