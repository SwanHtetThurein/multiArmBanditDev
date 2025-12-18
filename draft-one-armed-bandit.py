import numpy as np
import random
import math

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
#replace with actual reward generator logic or better logic to test different bandit strategies


def generate_beta_value(arm):
    random_beta_value = np.random.beta(arm[0], arm[1])
    return random_beta_value


def normalize_beta_values(beta_values):
    total = sum(beta_values)
    normalized_values = [value / total for value in beta_values]
    return normalized_values


def posterior_normalization(current_arm, nbv, bandit_type, current_round, total_rounds):

    if current_arm is None:
        return nbv
    
    pbv=nbv[::]
    if bandit_type == 'early':
        delta = 1/(1+math.e**(current_round - total_rounds/2))
    elif bandit_type == 'late':
        delta = 1/(1+math.e**(total_rounds/2 - current_round))
    else:
        delta = 1

    accumulator = 0

    print(f"Delta value: {delta}")

    for i in range(len(pbv)):
        if i != current_arm:
            accumulator += pbv[i]*(1-delta)
            pbv[i] = pbv[i]*(delta)
    
    pbv[current_arm] +=  accumulator

    return pbv


def main():

    total_rounds = int(input("Enter number of rounds to simulate: "))
    arm_chosen = None
    for i in range(total_rounds):
        print(f"Round {i+1} \n\n")

        beta_values = [generate_beta_value(centralized), generate_beta_value(decentralized), generate_beta_value(no_heirarchy)]
        normed_beta_values = normalize_beta_values(beta_values)
        posterior_beta_values = posterior_normalization(arm_chosen, normed_beta_values, 'early', i, total_rounds)
        arm_chosen = np.random.choice([0, 1, 2], p=posterior_beta_values)

        print(f"Centralized: {centralized}, Decentralized: {decentralized}, No Heirarchy: {no_heirarchy}\n")
        print(f"Beta values - Centralized: {beta_values[0]}, Decentralized: {beta_values[1]}, No Heirarchy: {beta_values[2]}\n")
        print(f"Normalized Beta values - Centralized: {normed_beta_values[0]}, Decentralized: {normed_beta_values[1]}, No Heirarchy: {normed_beta_values[2]}\n")
        print(f"Posterior Beta values - Centralized: {posterior_beta_values[0]}, Decentralized: {posterior_beta_values[1]}, No Heirarchy: {posterior_beta_values[2]}\n")
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

#test comment
main()