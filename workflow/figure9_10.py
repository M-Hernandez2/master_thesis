#Author: Madison Hernandez
#Purpose: factor importance on final well salinization; use delta method of sensitivity analysis to understand factor control on model output and plot, creating figures 9 and 10

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from SALib.analyze import delta

scenarios = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SEAWAT_model_inputs\PT_mean_std_SLR.xlsx", sheet_name='full')

#dictionary with system parameters and bounds
problem = {'num_vars': 5,
           'names': ['μ precip', 'σ precip', 'μ temp', 'σ temp', 'SLR'],
           'bounds': [[123, 2504],
                      [54, 399],
                      [10, 22],
                      [0.3, 2.5],
                      [0.4, 2.5]]}


#we have 500 samples , used Latin Hypercube method, imported with 'scenarios'
x_set1 = scenarios.iloc[:, 1:6]
output = scenarios.iloc[:, 6:]
res_dict = {}

#names of the 16 wells to do factor importance on
output_names = ['2718', '2723', '2727', '2732', '2759', '2761', '2762', '2799', '2803', '2804', '2805', '2806', '2940', '2943', '2953', '2963']
input_names = problem['names']

delta_results = []
delta_conf_results = []

#convert output column into a 1d array for the model
for col in output.columns:
    Y = output[col].values

    #perform delta analysis
    results = delta.analyze(problem, np.asarray(x_set1), Y, print_to_console=True)
    delta_results.append(results['delta'])
    delta_conf_results.append(results['delta_conf'])

delta_array = np.array(delta_results).T
delta_conf_array = np.array(delta_conf_results).T

#create dataframe of results
sensitivity_df = pd.DataFrame(delta_array, index=input_names, columns=output_names)

#sort factors by importance
factors_sorted = np.argsort(results['delta'])[::-1]
print(factors_sorted)


#visualize factor importance
mean_delta = np.mean(delta_array, axis=1)
mean_delta_conf = np.mean(delta_conf_array, axis=1)

#sort factors by importance
sorted_indices = np.argsort(mean_delta)[::-1]  #descending order
sorted_factors = [input_names[i] for i in sorted_indices]
sorted_delta = mean_delta[sorted_indices]
sorted_conf = mean_delta_conf[sorted_indices]
sorted_delta_data = delta_array[sorted_indices]



#data to create and customize boxplot for delta indexes
plt.figure(figsize=(8,5))
boxplot = plt.boxplot(sorted_delta_data.T, labels=sorted_factors, patch_artist=True)
for box in boxplot['boxes']:
    box.set_facecolor('maroon')
    box.set_alpha(0.8)
    box.set_edgecolor('navy')
for whisker in boxplot['whiskers']:
    whisker.set(color='navy')
    whisker.set_alpha(0.8)
for cap in boxplot['caps']:
    cap.set(color='navy')
    cap.set_alpha(0.8)
for median in boxplot['medians']:
    median.set(color='navy', linewidth=2, alpha=0.8)
for flier in boxplot['fliers']:
    flier.set(marker='o', color='navy', alpha=0.6)
plt.ylabel('Delta Index')
plt.xlabel('Input Variables')
plt.ylim([0.0, 0.30])
plt.title('Factor Prioritization (Delta Method)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#sort in order of descinding factor importance
for output_name, result in res_dict.items():
    #sort in descending order
    sorted_indices = np.argsort(result['delta'])[::-1]
    sorted_deltas = result['delta'][sorted_indices]
    sorted_conf = result['delta_conf'][sorted_indices]
    sorted_names = np.array(result['names'])[sorted_indices]


#set colors and bounds for following heatmap
bounds = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
colors = ['lavenderblush', 'lightpink', 'salmon', 'indianred', 'maroon']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

#plot heatmap showing each wells delata sensitivity index
plt.figure(figsize=(12, 8))
sns.heatmap(sorted_delta_data, annot=True, cmap=cmap, norm=norm, square=True, linewidths=0.5, fmt=".2f")

plt.title('Delta Sensitivity Heatmap for Model Outputs', fontsize=16)
plt.xlabel('Model Outputs', fontsize=12)
plt.ylabel('Input Parameters', fontsize=12)
plt.tight_layout()
plt.show()
