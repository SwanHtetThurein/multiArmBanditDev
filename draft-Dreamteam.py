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

pre_global_post_nbv = [[None,[None,None,None]],[None,[None,None,None]],[None,[None,None,None]],[None,[None,None,None,None,None]],[None,[None,None,None]]]


def reward_generator():
    t = random.random()
    if t>0.5:
        return 1
    return 0
#replace with actual reward generator logic or better logic to test different bandit strategies

#pick random sample value from beta distribution 
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

    for i in range(len(pbv)):
        if i != current_arm:
            accumulator += pbv[i]*(1-delta)
            pbv[i] = pbv[i]*(delta)
    
    pbv[current_arm] +=  accumulator

    return pbv


def global_contraint(pre_global_post_nbv, current_round, total_rounds, m=3, d=5):

    post_global_post_nbv = pre_global_post_nbv.copy()

    y = m * (1 - ( (current_round - (total_rounds/2)) / (total_rounds/2) ) **2)
    print(f"Y value: {y} \n")

    z = 0
    zd_list = []

    for bandit in pre_global_post_nbv:
        a = sum(bandit[1])
        a -= bandit[1][bandit[0]]
        zd_list.append(a)

    print("zd_list: ", zd_list)
    z = sum(zd_list)
    print(f"Z value: {z}")
    excess = z-y
    print
    if excess <= 0:
        return pre_global_post_nbv
    
    for zd,bandit,ind in zip(zd_list,pre_global_post_nbv,range(len(pre_global_post_nbv))):
        
        delta2 =   max(0, 1 - (excess/(zd * d)))
        print(f"Delta2 value for bandit {ind}: {delta2}\n")

        curr_arm = bandit[0]

        s_p = 0
        for j in enumerate(bandit[1]):

            if j[0] != curr_arm:
                s_p += post_global_post_nbv[ind][1][j[0]] * (1 - delta2)
                post_global_post_nbv[ind][1][j[0]] = j[1] * delta2
 

        post_global_post_nbv[ind][1][curr_arm] += s_p

    return post_global_post_nbv


def main():

    total_rounds = int(input("Enter number of rounds to simulate: "))
    arms_chosens = [None,None,None,None,None]
    for i in range(total_rounds):
        print(f"Round {i+1} \n\n")

        for j,Bandit in enumerate(Bandits):
            arm_chosen = arms_chosens[j]
            
            #Bandit[0].values gives the arms of the bandit
            beta_values = [generate_beta_value(x[0]) for x in Bandit[0].values()]
            normed_beta_values = normalize_beta_values(beta_values)

            posterior_beta_values = posterior_normalization(arm_chosen, normed_beta_values, Bandit[1], i+1, total_rounds)

            pre_global_post_nbv[j][0] = arm_chosen
            pre_global_post_nbv[j][1] = posterior_beta_values


            #select arm based on posterior probabilities for first round
            if i == 0:
                arm_chosen = int(np.random.choice(list(Bandit[0].keys()), p=posterior_beta_values))
                arms_chosens[j]=arm_chosen
                continue
        print("Pre-global posterior nbv: ",pre_global_post_nbv,"\n")

        #Integrating Global constraints from second round onwards
        if i >= 1:
            
            post_global_post_nbv = global_contraint(pre_global_post_nbv, i+1, total_rounds)
            print("Post-global posterior nbv: ",post_global_post_nbv,"\n")

            for arm_ind,Bandit in zip(range(len(arms_chosens)),Bandits):
                
                arm_chosen = int(np.random.choice(list(Bandit[0].keys()), p=post_global_post_nbv[arm_ind][1]))
                arms_chosens[arm_ind]=arm_chosen


        #print(f"Arms chosen for this round: {arms_chosens}\n")
        print([x[1][0][arms_chosens[x[0]]][1] for x in enumerate(Bandits)])


        #if the combination of arms chosen results in success or failure
        sucess = reward_generator()

        if sucess:
            print("Success\n")
            for Bandit, arm in zip(Bandits, arms_chosens):
                #update the beta distribution parameters for the chosen arms
                Bandit[0][arm][0][0] += 1
        
        else:
            print("Failed\n")
            for Bandit, arm in zip(Bandits, arms_chosens):
                #update the beta distribution parameters for the chosen arms
                Bandit[0][arm][0][1] += 1
        
    print(Heirarchy)
    print(Interaction_patterns)
    print(Norms_of_Engagement)
    print(Decision_making_norms)
    print(Feedback_norms)

main()