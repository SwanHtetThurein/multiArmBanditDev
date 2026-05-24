import numpy as np
import random
import math

class five_bandit():
    def __init__(self, initial_bias, optimal_arm, trounds):
        
        self.Heirarchy = {
            0: [[1, 1], 'Centralized'],
            1: [[1, 1], 'Decentralized'],
            2: [[1, 1], 'No-heirarchy']
        }

        self.Interaction_patterns = {
            0:[[1,1],'Emergent'],
            1:[[1,1],'Round-robin'],
            2:[[1,1],'Equally-distributed']
        }

        self.Norms_of_Engagement = {            
            0:[[1,1],'None'],
            1:[[1,1],'Professional'],
            2:[[1,1],'Informal']
        }

        self.Decision_making_norms = {
            0:[[1,1],'None'],
            1:[[1,1],'Divergent'],
            2:[[1,1],'Convergent'],
            3:[[1,1],'Informed'],
            4:[[1,1],'Rapid']
        }

        self.Feedback_norms = {
            0:[[1,1],'None'],
            1:[[1,1],'Encouraging'],
            2:[[1,1],'Critical'],
        }



        self.dict_list = [self.Heirarchy, self.Interaction_patterns, self.Norms_of_Engagement, self.Decision_making_norms, self.Feedback_norms]
        self.initial_bias = initial_bias
        self.optimal_arm = optimal_arm
        self.total_rounds = trounds

    def reward_generator(self, curr_arms, optimal_arms, noise=0.175):
        """
        noise = probability of a *wrong* answer when arms are optimal,
                or probability of a *right* answer when arms are wrong.
        """
        p = 1 
        for chosen, opt in zip(curr_arms, optimal_arms):
            if chosen != opt:
                p -= 1/(len(curr_arms))     # penalise an incorrect choice
            # else:
            #     p += noise * 0.2    # small bonus for correct arm

        # p = max(0.0, min(1.0, p))   # clamp to [0,1]
        return p

    def generate_beta_value(self, arm):
        random_beta_value = np.random.beta(arm[0], arm[1])
        return random_beta_value

    def calculate_beta_mean(self, arm):
        alpha, beta = arm[0], arm[1]
        mean_beta_value = (alpha) / (alpha + beta )
        return mean_beta_value


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
            delta = 1/(1+math.e**(current_round- (total_rounds/2) ))
        elif bandit_type == 'late':
            # Late rounds: exploit more (higher delta = more concentration)
            delta = 1/(1+math.e**( (total_rounds/2) - current_round))
        else:
            delta = 1

        accumulator = 0

        print(f"Delta value: {delta}")

        for i in range(len(pbv)):
            if i != current_arm:
                accumulator += pbv[i]*(1-delta)
                pbv[i] = pbv[i] * delta
        
        pbv[current_arm] +=  accumulator

        return pbv


    def main(self):
        total_rounds = self.total_rounds
        arms_chosen = self.initial_bias
        for i in range(total_rounds):

            print(f"Round {i+1} \n\n")
            print(f"Current arm right now: {arms_chosen}\n")                


            Hierarchy_beta_values = [self.generate_beta_value(self.Heirarchy[j][0]) for j in range(3)]
            Interaction_beta_values = [self.generate_beta_value(self.Interaction_patterns[j][0]) for j in range(3)]
            Norms_of_Engagement_beta_values = [self.generate_beta_value(self.Norms_of_Engagement[j][0]) for j in range(3)]
            Decision_making_norms_beta_values = [self.generate_beta_value(self.Decision_making_norms[j][0]) for j in range(5)]
            Feedback_norms_beta_values = [self.generate_beta_value(self.Feedback_norms[j][0]) for j in range(3)]

            Hierarchy_normed_beta_values = self.normalize_beta_values(Hierarchy_beta_values)
            Interaction_normed_beta_values = self.normalize_beta_values(Interaction_beta_values)
            Norms_of_Engagement_normed_beta_values = self.normalize_beta_values(Norms_of_Engagement_beta_values)
            Decision_making_norms_normed_beta_values = self.normalize_beta_values(Decision_making_norms_beta_values)
            Feedback_norms_normed_beta_values = self.normalize_beta_values(Feedback_norms_beta_values)

            Hierarchy_posterior_beta_values = self.posterior_normalization(arms_chosen[0], Hierarchy_normed_beta_values, 'early', i+1, total_rounds)
            Interaction_posterior_beta_values = self.posterior_normalization(arms_chosen[1], Interaction_normed_beta_values, 'ongoing', i+1, total_rounds)
            Norms_of_Engagement_posterior_beta_values = self.posterior_normalization(arms_chosen[2], Norms_of_Engagement_normed_beta_values, 'ongoing', i+1, total_rounds)
            Decision_making_norms_posterior_beta_values = self.posterior_normalization(arms_chosen[3], Decision_making_norms_normed_beta_values, 'ongoing', i+1, total_rounds)
            Feedback_norms_posterior_beta_values = self.posterior_normalization(arms_chosen[4], Feedback_norms_normed_beta_values, 'ongoing', i+1, total_rounds)
            
            if i == 0:
                print("First round, choosing arms based on initial bias")
                arms_chosen = self.initial_bias
            else:
                arms_chosen[0] = np.random.choice([0, 1, 2], p=Hierarchy_posterior_beta_values)
                arms_chosen[1] = np.random.choice([0, 1, 2], p=Interaction_posterior_beta_values)
                arms_chosen[2] = np.random.choice([0, 1, 2], p=Norms_of_Engagement_posterior_beta_values)
                arms_chosen[3] = np.random.choice([0, 1, 2, 3, 4], p=Decision_making_norms_posterior_beta_values)
                arms_chosen[4] = np.random.choice([0, 1, 2], p=Feedback_norms_posterior_beta_values)

            print("For Hierarchy:")
            print(f"Centralized: {self.Heirarchy[0][0]}, Decentralized: {self.Heirarchy[1][0]}, No Heirarchy: {self.Heirarchy[2][0]}\n")
            print(f"Beta values - Centralized: {Hierarchy_beta_values[0]}, Decentralized: {Hierarchy_beta_values[1]}, No Heirarchy: {Hierarchy_beta_values[2]}\n")
            print(f"Normalized Beta values - Centralized: {Hierarchy_normed_beta_values[0]}, Decentralized: {Hierarchy_normed_beta_values[1]}, No Heirarchy: {Hierarchy_normed_beta_values[2]}\n")
            print(f"Posterior Beta values - Centralized: {Hierarchy_posterior_beta_values[0]}, Decentralized: {Hierarchy_posterior_beta_values[1]}, No Heirarchy: {Hierarchy_posterior_beta_values[2]}\n\n")

            print("For Interaction patterns:")
            print(f"Emergent: {self.Interaction_patterns[0][0]}, Round-robin: {self.Interaction_patterns[1][0]}, Equally-distributed: {self.Interaction_patterns[2][0]}\n")
            print(f"Beta values - Emergent: {Interaction_beta_values[0]}, Round-robin: {Interaction_beta_values[1]}, Equally-distributed: {Interaction_beta_values[2]}\n")
            print(f"Normalized Beta values - Emergent: {Interaction_normed_beta_values[0]}, Round-robin: {Interaction_normed_beta_values[1]}, Equally-distributed: {Interaction_normed_beta_values[2]}\n")
            print(f"Posterior Beta values - Emergent: {Interaction_posterior_beta_values[0]}, Round-robin: {Interaction_posterior_beta_values[1]}, Equally-distributed: {Interaction_posterior_beta_values[2]}\n\n")


            print("For Norms of Engagement:")
            print(f"High: {self.Norms_of_Engagement[0][0]}, Medium: {self.Norms_of_Engagement[1][0]}, Low: {self.Norms_of_Engagement[2][0]}\n")
            print(f"Beta values - High: {Norms_of_Engagement_beta_values[0]}, Medium: {Norms_of_Engagement_beta_values[1]}, Low: {Norms_of_Engagement_beta_values[2]}\n")
            print(f"Normalized Beta values - High: {Norms_of_Engagement_normed_beta_values[0]}, Medium: {Norms_of_Engagement_normed_beta_values[1]}, Low: {Norms_of_Engagement_normed_beta_values[2]}\n")
            print(f"Posterior Beta values - High: {Norms_of_Engagement_posterior_beta_values[0]}, Medium: {Norms_of_Engagement_posterior_beta_values[1]}, Low: {Norms_of_Engagement_posterior_beta_values[2]}\n\n")


            print("For Decision-making norms:")
            print(f"None: {self.Decision_making_norms[0][0]}, Divergent: {self.Decision_making_norms[1][0]}, Convergent: {self.Decision_making_norms[2][0]}, Informed: {self.Decision_making_norms[3][0]}, Rapid: {self.Decision_making_norms[4][0]}\n")
            print(f"Beta values - None: {Decision_making_norms_beta_values[0]}, Divergent: {Decision_making_norms_beta_values[1]}, Convergent: {Decision_making_norms_beta_values[2]}, Informed: {Decision_making_norms_beta_values[3]}, Rapid: {Decision_making_norms_beta_values[4]}\n")      
            print(f"Normalized Beta values - None: {Decision_making_norms_normed_beta_values[0]}, Divergent: {Decision_making_norms_normed_beta_values[1]}, Convergent: {Decision_making_norms_normed_beta_values[2]}, Informed: {Decision_making_norms_normed_beta_values[3]}, Rapid: {Decision_making_norms_normed_beta_values[4]}\n")
            print(f"Posterior Beta values - None: {Decision_making_norms_posterior_beta_values[0]}, Divergent: {Decision_making_norms_posterior_beta_values[1]}, Convergent: {Decision_making_norms_posterior_beta_values[2]}, Informed: {Decision_making_norms_posterior_beta_values[3]}, Rapid: {Decision_making_norms_posterior_beta_values[4]}\n\n")


            print("For Feedback norms:")
            print(f"None: {self.Feedback_norms[0][0]}, Encouraging {self.Feedback_norms[1][0]}, Critical: {self.Feedback_norms[2][0]}\n")
            print(f"Beta values - None: {Feedback_norms_beta_values[0]}, Encouraging: {Feedback_norms_beta_values[1]}, Critical: {Feedback_norms_beta_values[2]}\n")
            print(f"Normalized Beta values - None: {Feedback_norms_normed_beta_values[0]}, Encouraging: {Feedback_norms_normed_beta_values[1]}, Critical: {Feedback_norms_normed_beta_values[2]}\n")
            print(f"Posterior Beta values - None: {Feedback_norms_posterior_beta_values[0]}, Encouraging: {Feedback_norms_posterior_beta_values[1]}, Critical: {Feedback_norms_posterior_beta_values[2]}\n")




            print("Arms chosen for this round:")
            print(f"Chosen arms: {self.Heirarchy[arms_chosen[0]][1]}, {self.Interaction_patterns[arms_chosen[1]][1]}, {self.Norms_of_Engagement[arms_chosen[2]][1]}, {self.Decision_making_norms[arms_chosen[3]][1]}, {self.Feedback_norms[arms_chosen[4]][1]}\n")

            success = self.reward_generator(arms_chosen, self.optimal_arm)
            print(f"Success: {success}")
            if success > 0.1:
                print("Success")
                self.Heirarchy[arms_chosen[0]][0][0] += success
                self.Interaction_patterns[arms_chosen[1]][0][0] += success
                self.Norms_of_Engagement[arms_chosen[2]][0][0] += success
                self.Decision_making_norms[arms_chosen[3]][0][0] += success
                self.Feedback_norms[arms_chosen[4]][0][0] += success
            else:
                print("Failed")
                self.Heirarchy[arms_chosen[0]][0][1] += 1000
                self.Interaction_patterns[arms_chosen[1]][0][1] += 1000
                self.Norms_of_Engagement[arms_chosen[2]][0][1] += 1000
                self.Decision_making_norms[arms_chosen[3]][0][1] += 1000
                self.Feedback_norms[arms_chosen[4]][0][1] += 1000

            #Beta Distribution = B (alpha, beta) where alpha = number of successes + 1 and beta = number of failures + 1. So we add 1 to both success and failure counts to avoid issues with zero counts.



        Hierarchy_mean_beta_values = [
            self.calculate_beta_mean(self.Heirarchy[0][0]),
            self.calculate_beta_mean(self.Heirarchy[1][0]),
            self.calculate_beta_mean(self.Heirarchy[2][0])
        ]


        Interaction_mean_beta_values = [
            self.calculate_beta_mean(self.Interaction_patterns[0][0]),
            self.calculate_beta_mean(self.Interaction_patterns[1][0]),
            self.calculate_beta_mean(self.Interaction_patterns[2][0])
        ]

        Norms_of_Engagement_mean_beta_values = [
            self.calculate_beta_mean(self.Norms_of_Engagement[0][0]),
            self.calculate_beta_mean(self.Norms_of_Engagement[1][0]),
            self.calculate_beta_mean(self.Norms_of_Engagement[2][0])
        ]

        Decision_making_norms_mean_beta_values = [
            self.calculate_beta_mean(self.Decision_making_norms[0][0]),
            self.calculate_beta_mean(self.Decision_making_norms[1][0]),
            self.calculate_beta_mean(self.Decision_making_norms[2][0]),
            self.calculate_beta_mean(self.Decision_making_norms[3][0]),
            self.calculate_beta_mean(self.Decision_making_norms[4][0])
        ]


        Feedback_norms_mean_beta_values = [
            self.calculate_beta_mean(self.Feedback_norms[0][0]),
            self.calculate_beta_mean(self.Feedback_norms[1][0]),
            self.calculate_beta_mean(self.Feedback_norms[2][0])
        ]

        print(f"Final updated Hierarchy mean beta values: Centralized: {Hierarchy_mean_beta_values[0]}, Decentralized: {Hierarchy_mean_beta_values[1]}, No Heirarchy: {Hierarchy_mean_beta_values[2]}")
        print(f"Final updated Interaction mean beta values: Emergent: {Interaction_mean_beta_values[0]}, Round-robin: {Interaction_mean_beta_values[1]}, Equally-distributed: {Interaction_mean_beta_values[2]}")
        print(f"Final updated Norms of Engagement mean beta values: {Norms_of_Engagement_mean_beta_values[0]}, {Norms_of_Engagement_mean_beta_values[1]}, {Norms_of_Engagement_mean_beta_values[2]}")
        print(f"Final updated Decision Making norms mean beta values: {Decision_making_norms_mean_beta_values[0]}, {Decision_making_norms_mean_beta_values[1]}, {Decision_making_norms_mean_beta_values[2]}, {Decision_making_norms_mean_beta_values[3]}, {Decision_making_norms_mean_beta_values[4]}")
        print(f"Final updated Feedback norms mean beta values: {Feedback_norms_mean_beta_values[0]}, {Feedback_norms_mean_beta_values[1]}, {Feedback_norms_mean_beta_values[2]}")

        Hierarchy_max_arm_index = np.argmax(Hierarchy_mean_beta_values)
        Interaction_max_arm_index = np.argmax(Interaction_mean_beta_values)
        Norms_of_Engagement_max_arm_index = np.argmax(Norms_of_Engagement_mean_beta_values)
        Decision_making_norms_max_arm_index = np.argmax(Decision_making_norms_mean_beta_values)
        Feedback_norms_max_arm_index = np.argmax(Feedback_norms_mean_beta_values)


        count = 0

        if Hierarchy_max_arm_index == self.optimal_arm[0]:
            count += 1
        if Interaction_max_arm_index == self.optimal_arm[1]:
            count += 1
        if Norms_of_Engagement_max_arm_index == self.optimal_arm[2]:
            count += 1
        if Decision_making_norms_max_arm_index == self.optimal_arm[3]:
            count += 1
        if Feedback_norms_max_arm_index == self.optimal_arm[4]:
            count += 1

        return count
    
tests = [
    [[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,1,1,1],[0,1,1,1,1]],
    [[0,1,2,2,2],[1,2,2,2,2]],
    [[1,2,2,2,2],[2,2,2,2,2]],
    [[1,2,1,1,1],[2,1,1,1,1]],
    [[0,1,0,0,0],[1,0,0,0,0]],
    [[0,0,0,0,0],[2,2,2,2,2]],
    [[1,1,1,1,1],[1,1,1,1,1]],
    [[2,2,2,2,2],[0,0,0,0,0]]
]


d={
    10:[0,0,0],
    # 15:[0,0,0],
    # 20:[0,0,0]
}

for j in [10,15,20]:
    success_count = 0
    partial_success_count = 0
    failure_count = 0

    score = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0

    }

    for i in tests*20:
        print(f"Initial bias: {i[0]}, Optimal arm: {i[1]}")
        bandit = five_bandit(i[0], i[1], j)
        result = bandit.main()
        score[result] +=1

    d[j] = f"Successes: {score[5]}, Failures: {score[0]}, Partial Successes: 1 - {score[1]}, 2 - {score[2]}, 3 - {score[3]}, 4 - {score[4]}"


for key, value in d.items():
    print(f"Total rounds: {key}, {value}")

    