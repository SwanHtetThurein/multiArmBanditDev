import numpy as np
import random
import math

# index_to_arm = {
#     0:'centralized',
#     1:'decentralized',
#     2:'no_heirarchy'
# }

# centralized = [1,1]

# decentralized = [1,1]

# no_heirarchy = [1,1]



class one_armed_bandit():



    def __init__(self, initial_bias, optimal_arm, trounds):
        
        self.Heirarchy = {
            0: [[1, 1], 'Centralized'],
            1: [[1, 1], 'Decentralized'],
            2: [[1, 1], 'No-heirarchy']
        }


        self.initial_bias = initial_bias
        self.optimal_arm = optimal_arm
        self.total_rounds = trounds


    def reward_generator(self,curr_arm, optimal_arm):
        base = 0.001
        
        if curr_arm == optimal_arm:
            base += 0.998
        
        return 1 if random.random() < base else 0


    def generate_beta_value(self, arm):
        random_beta_value = np.random.beta(arm[0], arm[1])
        return random_beta_value


    def normalize_beta_values(self,beta_values):
        total = sum(beta_values)
        normalized_values = [value / total for value in beta_values]
        return normalized_values


    def posterior_normalization(self, current_arm, nbv, bandit_type, current_round, total_rounds):

        if current_arm is None:
            return nbv
        
        pbv=nbv[::]
        if bandit_type == 'early':
            # Early rounds: explore more (lower delta = less concentration)
            delta = 1/(1+math.e**(current_round-total_rounds/2))
        elif bandit_type == 'late':
            # Late rounds: exploit more (higher delta = more concentration)
            delta = 1/(1+math.e**(total_rounds/2-current_round))
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


    def main(self):

        #total_rounds = int(input("Enter number of rounds to simulate: "))
        total_rounds = self.total_rounds
        arm_chosen = self.initial_bias
        for i in range(total_rounds):

            print(f"Round {i+1} \n\n")
            print(f"Current arm right now: {arm_chosen}\n")

            beta_values = [self.generate_beta_value(self.Heirarchy[j][0]) for j in range(3)]
            normed_beta_values = self.normalize_beta_values(beta_values)
            posterior_beta_values = self.posterior_normalization(arm_chosen, normed_beta_values, 'early', i+1, total_rounds)

            if i == 0:
                print("First round, choosing arm based on initial bias")
                arm_chosen = self.initial_bias
            else:
                arm_chosen = np.random.choice([0, 1, 2], p=posterior_beta_values)

            print(f"Centralized: {self.Heirarchy[0][0]}, Decentralized: {self.Heirarchy[1][0]}, No Heirarchy: {self.Heirarchy[2][0]}\n")
            print(f"Beta values - Centralized: {beta_values[0]}, Decentralized: {beta_values[1]}, No Heirarchy: {beta_values[2]}\n")
            print(f"Normalized Beta values - Centralized: {normed_beta_values[0]}, Decentralized: {normed_beta_values[1]}, No Heirarchy: {normed_beta_values[2]}\n")
            print(f"Posterior Beta values - Centralized: {posterior_beta_values[0]}, Decentralized: {posterior_beta_values[1]}, No Heirarchy: {posterior_beta_values[2]}\n")
            print("Arm chosen for this round:")

            if arm_chosen == 0:
                print("Centralized")

                success = self.reward_generator(arm_chosen, self.optimal_arm)
                if success:
                    print("Success")
                    self.Heirarchy[0][0][0] += 1
                else:
                    print("Failed")
                    self.Heirarchy[0][0][1] += 1

            elif arm_chosen == 1:
                print("Decentralized")
                success = self.reward_generator(arm_chosen, self.optimal_arm)

                if success:
                    print("Success")
                    self.Heirarchy[1][0][0] += 1
                else:
                    print("Failed")
                    self.Heirarchy[1][0][1] += 1

            else:
                print("No Heirarchy")
                success = self.reward_generator(arm_chosen, self.optimal_arm)

                if success:
                    print("Success")
                    self.Heirarchy[2][0][0] += 1
                else:
                    print("Failed")
                    self.Heirarchy[2][0][1] += 1
            
            print("\n\n\n")


        beta_values = [
            self.generate_beta_value(self.Heirarchy[0][0]),
            self.generate_beta_value(self.Heirarchy[1][0]),
            self.generate_beta_value(self.Heirarchy[2][0])
        ]

        print(f"Final updated beta values: Centralized: {beta_values[0]}, Decentralized: {beta_values[1]}, No Heirarchy: {beta_values[2]}")

        max_arm_index = np.argmax(beta_values)

        return 1 if arm_chosen == max_arm_index else 0





    # Get a single random value
    #random_value_numpy = np.random.beta(a, b)
    #print("Single random value (NumPy):", random_value_numpy)

    # Get an array of 10 random values
    # sample_of_values = np.random.beta(a, b, size=10)
    # print("Sample of 10 values (NumPy):", sample_of_values)

    #test comment




tests = [
    [0,2],
    [0,1],
    [1,0],
    [1,2],
    [2,0],
    [2,1],
    [0,0],
    [1,1],
    [2,2]
]

d={
    10:[0,0],
    15:[0,0],
    20:[0,0]

}


for j in [10,15,20,50,100]:
    success_count = 0
    failure_count = 0


    for i in tests*20:
        print(f"Initial bias: {i[0]}, Optimal arm: {i[1]}")
        bandit = one_armed_bandit(i[0], i[1], j)
        result = bandit.main()
        if result == 1:
            success_count += 1
        else:
            failure_count += 1


    d[j] = [success_count, failure_count]



for key, value in d.items():
    print(f"Total rounds: {key}, Successes: {value[0]}, Failures: {value[1]}")