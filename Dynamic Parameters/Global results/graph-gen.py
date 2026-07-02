from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# df = pd.read_csv('global_results_rounds100_dims6_tests9_20260607_231322.csv')
#global_results_rounds10_dims6_tests9_20260614_225205.csv
#global_results_rounds50_dims6_tests9_20260614_225348.csv


# df = pd.read_csv('global_results_rounds100_dims9_tests9_20260607_231226.csv')
#global_results_rounds50_dims9_tests9_20260614_225524.csv
#global_results_rounds10_dims9_tests9_20260614_225513.csv


# df = pd.read_csv('global_results_rounds100_dims3_tests9_20260607_231307.csv')
#global_results_rounds10_dims3_tests9_20260614_224636.csv
#global_results_rounds50_dims3_tests9_20260614_224728.csv

file_name = 'global_results_rounds50_dims9_tests9_20260622_000118.csv'
csv_path = Path(__file__).resolve().with_name(file_name)
df = pd.read_csv(csv_path)

file_stem = Path(file_name).stem
dimensions_match = re.search(r'dims(\d+)', file_stem)
dimensions = dimensions_match.group(1) if dimensions_match else 'unknown'
is_global = 'global' in file_stem.lower()
plot_label = f"{'Global' if is_global else 'Non-global'} results - {dimensions} dimensions"

sns.lineplot(data=df, x='round', y='performance', hue='noise_level', errorbar='ci')
plt.title(plot_label)
plt.xlabel('Rounds, t')
plt.ylabel('Ratio of dimensions that the probing algorithm got right')
plt.ylim(0.0, 1.0)
plt.yticks([i / 10 for i in range(11)])
plt.show()