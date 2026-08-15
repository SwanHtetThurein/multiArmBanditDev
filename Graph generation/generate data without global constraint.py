import numpy as np
import random
import math
import csv
import os

class five_bandit():
    def __init__(self, initial_bias, optimal_arm, trounds, noise = 0.0):
        
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
        self.noise = noise

    def reward_generator(self, curr_arms, optimal_arms):
        p = 1 
        for chosen, opt in zip(curr_arms, optimal_arms):
            if chosen != opt:
                p -= 1/(len(curr_arms)) 
        std = self.noise*p
        noisy_reward =  np.random.normal(p, std) 
        return np.clip(noisy_reward, 0, 1)
    


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


        for i in range(len(pbv)):
            if i != current_arm:
                accumulator += pbv[i]*(1-delta)
                pbv[i] = pbv[i] * delta
        
        pbv[current_arm] +=  accumulator

        return pbv


    def main(self):
        total_rounds = self.total_rounds
        arms_chosen = self.initial_bias
        round_results = []
        for i in range(total_rounds):


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
            Norms_of_Engagement_posterior_beta_values = self.posterior_normalization(arms_chosen[2], Norms_of_Engagement_normed_beta_values, 'late', i+1, total_rounds)
            Decision_making_norms_posterior_beta_values = self.posterior_normalization(arms_chosen[3], Decision_making_norms_normed_beta_values, 'late', i+1, total_rounds)
            Feedback_norms_posterior_beta_values = self.posterior_normalization(arms_chosen[4], Feedback_norms_normed_beta_values, 'ongoing', i+1, total_rounds)
            
            if i == 0:
                arms_chosen = self.initial_bias
            else:
                arms_chosen[0] = np.random.choice([0, 1, 2], p=Hierarchy_posterior_beta_values)
                arms_chosen[1] = np.random.choice([0, 1, 2], p=Interaction_posterior_beta_values)
                arms_chosen[2] = np.random.choice([0, 1, 2], p=Norms_of_Engagement_posterior_beta_values)
                arms_chosen[3] = np.random.choice([0, 1, 2, 3, 4], p=Decision_making_norms_posterior_beta_values)
                arms_chosen[4] = np.random.choice([0, 1, 2], p=Feedback_norms_posterior_beta_values)



            success = self.reward_generator(arms_chosen, self.optimal_arm)
            if success > 0.01:
                self.Heirarchy[arms_chosen[0]][0][0] += success
                self.Interaction_patterns[arms_chosen[1]][0][0] += success
                self.Norms_of_Engagement[arms_chosen[2]][0][0] += success
                self.Decision_making_norms[arms_chosen[3]][0][0] += success
                self.Feedback_norms[arms_chosen[4]][0][0] += success
            else:
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

            round_results.append(count)

        return round_results
    


tests = [
    [[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,1,1,1],[0,1,1,1,1]],
    [[0,1,2,2,2],[1,2,2,2,2]],
    [[1,2,2,2,2],[2,2,2,2,2]],
    [[1,2,1,1,1],[2,1,1,1,1]],
    [[0,1,0,0,0],[1,0,0,0,0]],
    [[0,0,0,0,0],[2,2,2,2,2]],
    [[1,1,1,1,1],[1,1,1,1,1]],
    [[2,2,2,2,2],[1,1,1,1,1]]
]

noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 0%, 10%, 80% noise
total_rounds = 100
runs_per_test = 20  # number of runs per test to get mean and CI

rows = []
rowid = 0
runID = 0

for noise in noise_levels:
    for _ in range(runs_per_test):
        for test in tests:
            initial_bias, optimal_arm = test[0][:], test[1][:]  # copy to avoid mutation
            bandit = five_bandit(initial_bias, optimal_arm, total_rounds, noise)
            round_results = bandit.main()
            
            for round_num, count in enumerate(round_results, start=1):
                performance = count / 5  # convert to % (0 to 1)
                rows.append([rowid, runID, noise, round_num, performance])
                rowid += 1
            
            runID += 1

with open('bandit_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['rowid', 'runID', 'noise_level', 'round', 'performance'])
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to bandit_results.csv")
