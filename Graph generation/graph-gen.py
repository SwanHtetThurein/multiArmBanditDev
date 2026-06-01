import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df = pd.read_csv('bandit_results.csv')
# df = pd.read_csv('global_bandit_results.csv')
df = pd.read_csv('dynamic_global_bandit_results.csv')

sns.lineplot(data=df, x='round', y='performance', hue='noise_level', errorbar='ci')
plt.xlabel('Rounds, t')
plt.ylabel('Ratio of dimensions that the probing algorithm got right')
plt.show()