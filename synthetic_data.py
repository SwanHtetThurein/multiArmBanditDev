import draft_Dreamteam as dt




synthetic_teams = [
    [
        {
            #biased dimensions
            'Hierarchy' : 0,
            'Interaction_patterns' : 1,
            'Norms_of_Engagement' : 2,
            'Decision_making_norms' : 3,
            'Feedback_norms' : 1
        },
        {
            #optimal dimensions
            'Hierarchy' : 2,
            'Interaction_patterns' : 0,
            'Norms_of_Engagement' : 1,
            'Decision_making_norms' : 0,
            'Feedback_norms' : 2
        }
    ],
    [
        {
            #biased dimensions
            'Hierarchy' : 1,
            'Interaction_patterns' : 2,
            'Norms_of_Engagement' : 0,
            'Decision_making_norms' : 4,
            'Feedback_norms' : 0
        },
        {
            #optimal dimensions
            'Hierarchy' : 2,
            'Interaction_patterns' : 1,
            'Norms_of_Engagement' : 1,
            'Decision_making_norms' : 2,
            'Feedback_norms' : 1
        }
    ],
    [
        {
            #biased dimensions
            'Hierarchy' : 0,
            'Interaction_patterns' : 0,
            'Norms_of_Engagement' : 1,
            'Decision_making_norms' : 1,
            'Feedback_norms' : 2
        },
        {
            #optimal dimensions
            'Hierarchy' : 1,
            'Interaction_patterns' : 2,
            'Norms_of_Engagement' : 2,
            'Decision_making_norms' : 3,
            'Feedback_norms' : 0
        }
    ],
    [
        {
            #biased dimensions
            'Hierarchy' : 2,
            'Interaction_patterns' : 1,
            'Norms_of_Engagement' : 0,
            'Decision_making_norms' : 0,
            'Feedback_norms' : 2
        },
        {
            #optimal dimensions
            'Hierarchy' : 0,
            'Interaction_patterns' : 2,
            'Norms_of_Engagement' : 1,
            'Decision_making_norms' : 4,
            'Feedback_norms' : 1
        }
    ],
    [
        {
            #biased dimensions
            'Hierarchy' : 1,
            'Interaction_patterns' : 0,
            'Norms_of_Engagement' : 2,
            'Decision_making_norms' : 2,
            'Feedback_norms' : 0
        },
        {
            #optimal dimensions
            'Hierarchy' : 2,
            'Interaction_patterns' : 1,
            'Norms_of_Engagement' : 0,
            'Decision_making_norms' : 3,
            'Feedback_norms' : 2
        }
    ]
]




def main():

    for team in synthetic_teams[0:1]:
        biased_arms = list(team[0].values())
        optimal_arms = list(team[1].values())

        print("\n\n\n\n\n\n")
        print(f"Biased arms: {biased_arms}")
        print(f"Optimal arms: {optimal_arms}")


        dt.DreamTeam(biased_arms, optimal_arms)


main()