#Author: Madison Hernandez
#Purpose: create heatmaps to show final well salinization and absolute change in well salinization to create figure 8

#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

#import data for well salinization for 16 vulnerable wells
data_reduce = pd.read_excel("../NEW_inputs/combined_output_update.xlsx", sheet_name='reduced')
well_reduce = pd.read_excel("../NEW_inputs/combined_output_update.xlsx", sheet_name='reduced_wells')

well_reduce = well_reduce.iloc[:, 0].tolist()
print(well_reduce)

#open up salinization data
result = []
for i, row in data_reduce.iterrows():
    row_name = row.iloc[0]
    non_empty = [val for val in row.iloc[1:] if pd.notna(val)]
    if not non_empty:
        continue
    result.append([row_name, non_empty])

#dataframe with the final timesteps output in kg/m3
result = pd.DataFrame(result, columns=['Model Run', 'Value'])

#change to mg/l
result['Value'] = result['Value'].apply(lambda data: [x * 1000 for x in data])

#set colors and bounds for plotting
bounds = [0, 40, 70, 140, 200, 250, 350, 500]
colors = ['white', 'lavenderblush', 'lightpink', 'salmon', 'indianred', 'maroon', 'black']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

#create a heatmap of the final chloride concentration data
value_array = np.array(result['Value'].tolist())
y_labels = result['Model Run'].tolist()
sns.heatmap(data=value_array, xticklabels=well_reduce, cmap=cmap, norm=norm, cbar_kws={'label': 'Cl- concentration', 'extend': 'max'})
plt.yticks(fontsize=3)
plt.xticks(fontsize=5)
plt.title('Final Salt Concentrations at Irrigation Wells')
plt.ylabel('SOW', fontsize=7)
plt.xlabel('Well ID', fontsize=7)
plt.show()


#calculate the difference in chloride concentration over the simulations runs (yr 75 - yr 0)
salt_diff = pd.DataFrame(columns=['Model', 'Value'])
salt_diff['Model'] = result['Model Run']
salt_diff['Value'] = result['Value'].apply(lambda data: [abs(x - 15) for x in data])

#make numbers have two digits after decimal point
def round_floats(cell):
    if isinstance(cell, list):
        return [round(x,2) for x in cell]
    return cell

salt_diff = salt_diff.applymap(round_floats)

#create heatmap showing the absolute difference between timstep 75 and the initial
value_array = np.array(salt_diff['Value'].tolist())
y_labels = salt_diff['Model'].tolist()
sns.heatmap(data=value_array, xticklabels=well_reduce, cmap=cmap, norm=norm, cbar_kws={'label': 'Cl- concentration', 'extend': 'max'})
plt.yticks(fontsize=3)
plt.xticks(fontsize=5)
plt.title('Absolute difference in Chloride Concentration Over 75 Years')
plt.ylabel('SOW', fontsize=7)
plt.xlabel('Well ID', fontsize=7)
plt.show()
