from pathlib import Path
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_name = 'non_global_results_rounds100_dims9_tests9_20260607_233729.csv'
csv_path = Path(__file__).resolve().with_name(file_name)
df = pd.read_csv(csv_path)

#df = pd.read_csv('non_global_results_rounds50_dims3_tests9_20260614_231501.csv')
#non_global_results_rounds10_dims3_tests9_20260614_231453.csv
#non_global_results_rounds50_dims3_tests9_20260614_231501.csv

# df = pd.read_csv('non_global_results_rounds100_dims6_tests9_20260607_233714.csv')
#non_global_results_rounds10_dims6_tests9_20260614_231700.csv
#non_global_results_rounds50_dims6_tests9_20260614_231708.csv

# df = pd.read_csv('non_global_results_rounds100_dims9_tests9_20260607_233729.csv')
#non_global_results_rounds10_dims9_tests9_20260614_231714.csv
#non_global_results_rounds50_dims9_tests9_20260614_231722.csv

file_stem = Path(file_name).stem
dimensions_match = re.search(r'dims(\d+)', file_stem)
dimensions = dimensions_match.group(1) if dimensions_match else 'unknown'

plot_label = f"Non-global results - {dimensions} dimensions"

sns.lineplot(data=df, x='round', y='performance', hue='noise_level', errorbar='ci')
plt.title(plot_label)
plt.xlabel('Rounds, t')
plt.ylabel('Ratio of dimensions that the probing algorithm got right')
plt.ylim(0.0, 1.0)
plt.yticks([i / 10 for i in range(11)])
plt.show()