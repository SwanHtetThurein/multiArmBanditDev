import numpy as np
import random
import math
import csv
import os
from datetime import datetime


class dynamic_bandit():
    def __init__(self, bandits, bandit_types, initial_bias, optimal_arm, trounds, noise):
        self.noise = noise
        self.bandits = bandits
        self.bandit_types = bandit_types
        self.initial_bias = initial_bias
        self.optimal_arm = optimal_arm
        self.total_rounds = trounds

    # def reward_generator(self, curr_arms, optimal_arms):
    #     p = 1
    #     for chosen, opt in zip(curr_arms, optimal_arms):
    #         if chosen != opt:
    #             p -= 1 / len(curr_arms)
    #     std = self.noise * p
    #     noisy_reward = np.random.normal(p, std)
    #     return np.clip(noisy_reward, 0, 1)

    def reward_generator(self, curr_arms, optimal_arms):
        mismatches = sum(1 for chosen, opt in zip(curr_arms, optimal_arms) if chosen != opt)
        p = 1 - (mismatches / len(curr_arms))
        std = self.noise * p
        noisy_reward = np.random.normal(p, std)
        return np.clip(noisy_reward, 0, 1)

    def generate_beta_value(self, arm):
        return np.random.beta(arm[0], arm[1])

    def calculate_beta_mean(self, arm):
        alpha, beta = arm[0], arm[1]
        return alpha / (alpha + beta)

    def normalize_beta_values(self, beta_values):
        total = sum(beta_values)
        return [v / total for v in beta_values]

    def posterior_normalization(self, current_arm, nbv, bandit_type, current_round, total_rounds):
        if current_arm is None:
            return nbv
        pbv = nbv[::]
        if bandit_type == 'early':
            delta = 1 / (1 + math.e ** (current_round - (total_rounds / 2)))
        elif bandit_type == 'late':
            delta = 1 / (1 + math.e ** ((total_rounds / 2) - current_round))
        else:
            delta = 1

        accumulator = 0
        for i in range(len(pbv)):
            if i != current_arm:
                accumulator += pbv[i] * (1 - delta)
                pbv[i] = pbv[i] * delta
        pbv[current_arm] += accumulator
        return pbv

    # def global_constraint(self, pre_global_post_nbv, current_round, total_rounds, m=2):
    #     post_global_post_nbv = [[bandit[0], bandit[1][:]] for bandit in pre_global_post_nbv]
    #     y = m * (1 - (((current_round - (total_rounds / 2)) / (total_rounds / 2)) ** 2))

    #     zd_list = []
    #     for bandit in pre_global_post_nbv:
    #         curr_arm = bandit[0]
    #         off_mass = sum(bandit[1]) - bandit[1][curr_arm]
    #         zd_list.append(off_mass)

    #     z = sum(zd_list)
    #     if z <= y:
    #         return post_global_post_nbv

    #     scale = y / z
    #     for ind, (zd, bandit) in enumerate(zip(zd_list, pre_global_post_nbv)):
    #         curr_arm = bandit[0]
    #         moved_prob = 0
    #         for arm_idx, prob in enumerate(bandit[1]):
    #             if arm_idx != curr_arm:
    #                 new_prob = prob * scale
    #                 moved_prob += prob - new_prob
    #                 post_global_post_nbv[ind][1][arm_idx] = new_prob
    #         post_global_post_nbv[ind][1][curr_arm] += moved_prob

    #     return post_global_post_nbv


    def global_constraint(self, pre_global_post_nbv, current_round, total_rounds, m=2):
        post_global_post_nbv = [[bandit[0], bandit[1][:]] for bandit in pre_global_post_nbv]
        
        y = m * (1 - (((current_round - (total_rounds / 2)) / (total_rounds / 2)) ** 2))
        y = max(y, 1.0)  # always allow at least 1 expected change
        
        zd_list = []
        for bandit in pre_global_post_nbv:
            curr_arm = bandit[0]
            off_mass = sum(bandit[1]) - bandit[1][curr_arm]
            zd_list.append(off_mass)

        z = sum(zd_list)
        if z <= y:
            return post_global_post_nbv

        scale = y / z
        for ind, (zd, bandit) in enumerate(zip(zd_list, pre_global_post_nbv)):
            curr_arm = bandit[0]
            moved_prob = 0
            for arm_idx, prob in enumerate(bandit[1]):
                if arm_idx != curr_arm:
                    new_prob = prob * scale
                    moved_prob += prob - new_prob
                    post_global_post_nbv[ind][1][arm_idx] = new_prob
            post_global_post_nbv[ind][1][curr_arm] += moved_prob

        return post_global_post_nbv

    def main(self):
        total_rounds = self.total_rounds
        arms_chosen = self.initial_bias[:]
        round_results = []
        bandit_names = list(self.bandits.keys())

        for i in range(total_rounds):

            # Generate and normalize beta values for each bandit
            all_normed = []
            for b_name in bandit_names:
                b = self.bandits[b_name]
                n_arms = len(b)
                beta_vals = [self.generate_beta_value(b[j][0]) for j in range(n_arms)]
                normed = self.normalize_beta_values(beta_vals)
                all_normed.append(normed)

            # Posterior normalization using user-defined bandit types
            all_posterior = []
            for idx, normed in enumerate(all_normed):
                btype = self.bandit_types[idx]
                posterior = self.posterior_normalization(arms_chosen[idx], normed, btype, i + 1, total_rounds)
                all_posterior.append(posterior)

            pre_global_post_nbv = [[arms_chosen[idx], all_posterior[idx]] for idx in range(len(bandit_names))]

            if i == 0:
                arms_chosen = self.initial_bias[:]
            else:
                post_global_post_nbv = self.global_constraint(pre_global_post_nbv, i + 1, total_rounds)
                for idx, b_name in enumerate(bandit_names):
                    n_arms = len(self.bandits[b_name])
                    arms_chosen[idx] = np.random.choice(range(n_arms), p=post_global_post_nbv[idx][1])

            success = self.reward_generator(arms_chosen, self.optimal_arm)
            for idx, b_name in enumerate(bandit_names):
                if success > 0.1:
                    self.bandits[b_name][arms_chosen[idx]][0][0] += success
                else:
                    self.bandits[b_name][arms_chosen[idx]][0][1] += 1000

            # Calculate means and argmax for each bandit
            count = 0
            for idx, b_name in enumerate(bandit_names):
                b = self.bandits[b_name]
                means = [self.calculate_beta_mean(b[j][0]) for j in range(len(b))]
                best_arm = np.argmax(means)
                if best_arm == self.optimal_arm[idx]:
                    count += 1

            round_results.append(count)

        return round_results


# ── Helper functions ──────────────────────────────────────────────────────────

def clone_bandits(template):
    """Deep copy the bandit template so each run starts fresh."""
    cloned = {}
    for b_name, arms in template.items():
        cloned[b_name] = {}
        for arm_idx, (params, arm_name) in arms.items():
            cloned[b_name][arm_idx] = [[params[0], params[1]], arm_name]
    return cloned


def generate_tests(arm_counts, n_tests=9):
    """Generate random (initial_bias, optimal_arm) test pairs."""
    tests = []
    for _ in range(n_tests):
        initial_bias = [random.randint(0, n - 1) for n in arm_counts]
        optimal_arm  = [random.randint(0, n - 1) for n in arm_counts]
        tests.append([initial_bias, optimal_arm])
    return tests


# ── User input ────────────────────────────────────────────────────────────────

def get_user_input():
    print("\n=== DreamTeam Bandit Configuration ===\n")

    # --- Number of bandits (must be divisible by 3) ---
    while True:
        n_bandits = int(input("How many bandits? (must be divisible by 3 for equal early/late/ongoing split): "))
        if n_bandits > 0 and n_bandits % 3 == 0:
            break
        print(f"  ✗ {n_bandits} is not divisible by 3. Please re-enter a valid number (e.g. 3, 6, 9, 12...).\n")

    # --- Number of rounds ---
    while True:
        total_rounds = int(input("How many rounds per experiment? "))
        if total_rounds > 0:
            break
        print("  ✗ Rounds must be a positive integer. Please try again.\n")

    # --- Assign bandit types equally: n/3 of each ---
    group_size = n_bandits // 3
    type_pool = ['early'] * group_size + ['late'] * group_size + ['ongoing'] * group_size
    random.shuffle(type_pool)

    # --- Build bandits with random arm counts (2–5) ---
    bandits_template = {}
    bandit_names     = []
    arm_counts       = []
    bandit_types     = []

    print(f"\nGenerating {n_bandits} bandits ({group_size} early, {group_size} late, {group_size} ongoing)...\n")

    for b in range(n_bandits):
        b_name   = f"bandit_{b + 1}"
        n_arms   = random.randint(2, 5)
        btype    = type_pool[b]

        bandit_names.append(b_name)
        arm_counts.append(n_arms)
        bandit_types.append(btype)

        bandits_template[b_name] = {}
        for a in range(n_arms):
            a_name = f"arm_{a + 1}"
            bandits_template[b_name][a] = [[1, 1], a_name]

        print(f"  {b_name}: type={btype}, arms={n_arms}")

    return bandits_template, bandit_names, arm_counts, bandit_types, n_bandits, total_rounds


# ── Main ──────────────────────────────────────────────────────────────────────

bandits_template, bandit_names, arm_counts, bandit_types, n_bandits, total_rounds = get_user_input()

n_tests      = 9
tests        = generate_tests(arm_counts, n_tests=n_tests)
noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
runs_per_test = 20

print(f"\nGenerated {len(tests)} tests:")
for t in tests:
    print(f"  Initial bias: {t[0]}, Optimal arm: {t[1]}")

rows  = []
rowid = 0
runID = 0

for noise in noise_levels:
    for _ in range(runs_per_test):
        for test in tests:
            initial_bias, optimal_arm = test[0][:], test[1][:]
            bandits = clone_bandits(bandits_template)
            bandit  = dynamic_bandit(bandits, bandit_types, initial_bias, optimal_arm, total_rounds, noise)
            round_results = bandit.main()

            for round_num, count in enumerate(round_results, start=1):
                performance = count / n_bandits
                rows.append([rowid, runID, noise, round_num, performance])
                rowid += 1

            runID += 1

# ── Save results with descriptive timestamped filename ────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename  = f"global_results_rounds{total_rounds}_dims{n_bandits}_tests{n_tests}_{timestamp}.csv"
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Test results global")
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, filename)

with open(file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['rowid', 'runID', 'noise_level', 'round', 'performance'])
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to '{file_path}'")