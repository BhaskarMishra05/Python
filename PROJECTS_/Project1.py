import random
print("Hello and Welcome ! to the random number gussing game.")
random_number= random.randint(0,100)
chances=9
guess_counter=0
while guess_counter<=chances:
    guess_counter+=1
    my_guess=int(input("Enter a number between 0 - 100: "))
    if my_guess < random_number:
        print("Guess is too low.")
    elif my_guess > random_number:
        print("Guess is too high.")
    else:
        print(f"Congrats you got it.The Number is {random_number}. It took you {guess_counter} tries.")
        break
    if(guess_counter>chances):
        print("You have exausted all your tries.")