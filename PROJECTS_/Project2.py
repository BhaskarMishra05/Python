import random
random_number=random.randint(1,100)
counter=9
guess=0
while guess<counter:
    guess+=1
    type_shii=int(input("Enter a number between 1-100: "))
    if type_shii < random_number:
        if (random_number-type_shii) <=10:
            print("Hot.")
        else:
            print("Cold.")
    elif type_shii>random_number:
        if (type_shii-random_number) <=10:
            print("Hot.")
        else:
            print("Cold.")
    elif type_shii==random_number:
        print(f"Delicious. It took you {guess} tries")
    elif guess > counter:
        print(f"Unfortunatly you failed.The number was {random_number}")
    