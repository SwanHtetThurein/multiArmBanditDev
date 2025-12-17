import numpy as np
import random


# index_to_arm = {
#     0:'centralized',
#     1:'decentralized',
#     2:'no_heirarchy'
# }

centralized = [1,1]

decentralized = [1,1]

no_heirarchy = [1,1]




def reward_generator():
    t = random.random()
    if t>0.5:
        return 1
    return 0



def generate_beta_value(arm):
    random_beta_value = np.random.beta(arm[0], arm[1])
    return random_beta_value



def main():


    for i in range(20):
        beta_values = [generate_beta_value(centralized), generate_beta_value(decentralized), generate_beta_value(no_heirarchy)]
        arm_chosen = beta_values.index(max(beta_values))

        print(f"Round {i+1}")
        print(f"Centralized: {centralized}, Decentralized: {decentralized}, No Heirarchy: {no_heirarchy}")
        print(f"Beta values - Centralized: {beta_values[0]}, Decentralized: {beta_values[1]}, No Heirarchy: {beta_values[2]}")
        print("Arm chosen for this round:")

        if arm_chosen == 0:
            print("Centralized")

            success = reward_generator()

            if success:
                print("Success")

                centralized[0] += 1
            else:
                print("Failed")
                centralized[1] += 1


        elif arm_chosen == 1:
            print("Decentralized")
            success = reward_generator()

            if success:
                print("Success")

                decentralized[0] += 1
            else:
                print("Failed")
                decentralized[1] += 1            

        else:
            print("No Heirarchy")
            success = reward_generator()

            if success:
                print("Success")

                no_heirarchy[0] += 1
            else:
                print("Failed")
                no_heirarchy[1] += 1
        
        print("\n\n\n\n\n\n")


    beta_values = [generate_beta_value(centralized), generate_beta_value(decentralized), generate_beta_value(no_heirarchy)]

    print(f"Final updated beta values: Centralized: {beta_values[0]}, Decentralized: {beta_values[1]}, No Heirarchy: {beta_values[2]}")







# Get a single random value
#random_value_numpy = np.random.beta(a, b)
#print("Single random value (NumPy):", random_value_numpy)

# Get an array of 10 random values
# sample_of_values = np.random.beta(a, b, size=10)
# print("Sample of 10 values (NumPy):", sample_of_values)


main()