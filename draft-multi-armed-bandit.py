import numpy as np
import random
import math

Heirarchy = {
    0:[[1,1],'Centralized'],
    1:[[1,1],'Decentralized'],
    2:[[1,1],'No-heirarchy']
}


Interaction_patterns = {
    0:[[1,1],'Emergent'],
    1:[[1,1],'Round-robin'],
    2:[[1,1],'Equally-distributed']
}

Norms_of_Engagement = {
    0:[[1,1],'None'],
    1:[[1,1],'Professional'],
    2:[[1,1],'Informal']
}


Decision_making_norms = {
    0:[[1,1],'None'],
    1:[[1,1],'Divergent'],
    2:[[1,1],'Convergent'],
    3:[[1,1],'Informed'],
    4:[[1,1],'Rapid']
}


Feedback_norms = {
    0:[[1,1],'None'],
    1:[[1,1],'Encouraging'],
    2:[[1,1],'Critical'],
}


Bandits = [[Heirarchy,'early'], [Interaction_patterns,'ongoing'], [Norms_of_Engagement,'late'], [Decision_making_norms,'late'], [Feedback_norms,'ongoing']]


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

    # print(f"Delta value: {delta}")

    for i in range(len(pbv)):
        if i != current_arm:
            accumulator += pbv[i]*(1-delta)
            pbv[i] = pbv[i]*(delta)
    
    pbv[current_arm] +=  accumulator

    return pbv


def main():

    total_rounds = int(input("Enter number of rounds to simulate: "))
    arm_chosens = [None,None,None,None,None]
    for i in range(total_rounds):
        print(f"Round {i+1} \n\n")


        for j,Bandit in enumerate(Bandits):
            arm_chosen = arm_chosens[j]
            
            beta_values = [generate_beta_value(x[0]) for x in Bandit[0].values()]


            normed_beta_values = normalize_beta_values(beta_values)

            posterior_beta_values = posterior_normalization(arm_chosen, normed_beta_values, Bandit[1], i, total_rounds)

            arm_chosen = np.random.choice(list(Bandit[0].keys()), p=posterior_beta_values)
            arm_chosens[j]=arm_chosen



        #print(f"Arms chosen for this round: {arm_chosens}\n")
        print([x[1][0][arm_chosens[x[0]]][1] for x in enumerate(Bandits)])


        sucess = reward_generator()




        if sucess:
            print("Success\n")

            for Bandit, arm in zip(Bandits, arm_chosens):
                Bandit[0][arm][0][0] += 1


        else:
            print("Failed\n")
            for Bandit, arm in zip(Bandits, arm_chosens):
                Bandit[0][arm][0][1] += 1

        


    print(Heirarchy)
    print(Interaction_patterns)
    print(Norms_of_Engagement)
    print(Decision_making_norms)
    print(Feedback_norms)








#test comment
main()