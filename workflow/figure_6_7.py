#Author: Madison Hernandez
#Purpose: process outputs of seawat concentrations to make maps of the difference in intial and final conditions
#shows years 25, 50, and 75 of the simulation as well as 10th, median, and 90th percentile in change inchloride concentration
#RUN WITH SHELL SCRIPT: figure_6_7.sh

import os
import shutil
import numpy as np
import pandas as pd
import flopy.utils.binaryfile as bf
import flopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

#create new text file with all the results of the 500 runs for the last time step
directory = "../NEW_inputs/Final_Runs/CONCENTRATIONS/"
output_file = "whole_cl_output.txt"

initial_conditions = {}
conditions_25_years = {}
conditions_50_years = {}
final_conditions = {}

#read .OBS files from output of running SEAWAT
with open(output_file, "w") as output:
    for file_name in os.listdir(directory):
        if file_name.startswith("RunSLR") and file_name.endswith(".OBS"):
            run_number = file_name.split("SLR")[1].split("P")[0]
            p_number = file_name.split("P")[1].split(".")[0]
            file_path = os.path.join(directory, file_name)
            
            #read data
            ucn = bf.UcnFile(file_path)
            times = ucn.get_times()  #get times in days
            
            #timestep 25 and 50 years 
            idx_25_years = np.argmin(np.abs(np.array(times) - 9125))
            idx_50_years = np.argmin(np.abs(np.array(times) - 18250))
            
            #initial timestep data
            init_data = ucn.get_data(kstpkper=ucn.get_kstpkper()[0]) * 1000
            init_data[init_data < 9] = np.nan
            for layer in range(init_data.shape[0]):
                if layer not in initial_conditions:
                    initial_conditions[layer] = [init_data[layer, :, :]]
                else:
                    initial_conditions[layer].append(init_data[layer, :, :])
            
            #25 years timestep data
            data_25_years = ucn.get_data(kstpkper=ucn.get_kstpkper()[idx_25_years]) * 1000
            data_25_years[data_25_years < 9] = np.nan
            for layer in range(data_25_years.shape[0]):
                if layer not in conditions_25_years:
                    conditions_25_years[layer] = [data_25_years[layer, :, :]]
                else:
                    conditions_25_years[layer].append(data_25_years[layer, :, :])
            
            #50 years timestep data
            data_50_years = ucn.get_data(kstpkper=ucn.get_kstpkper()[idx_50_years]) * 1000
            data_50_years[data_50_years < 9] = np.nan
            for layer in range(data_50_years.shape[0]):
                if layer not in conditions_50_years:
                    conditions_50_years[layer] = [data_50_years[layer, :, :]]
                else:
                    conditions_50_years[layer].append(data_50_years[layer, :, :])
            
            # final (year 75) timestep data
            final_data = ucn.get_data(kstpkper=ucn.get_kstpkper()[-1]) * 1000
            final_data[final_data < 9] = np.nan
            for layer in range(final_data.shape[0]):
                if layer not in final_conditions:
                    final_conditions[layer] = [final_data[layer, :, :]]
                else:
                    final_conditions[layer].append(final_data[layer, :, :])

#calculate differences for each timestep
def calculate_differences(initial_conditions, conditions, timestep_name):
    absolute_diff_median = {}
    absolute_diff_90th = {}
    absolute_diff_10th = {}
    
    for layer in initial_conditions:
        #median
        median_initial = np.nanmedian(np.stack(initial_conditions[layer], axis=0), axis=0)
        median_timestep = np.nanmedian(np.stack(conditions[layer], axis=0), axis=0)
        absolute_diff_median[layer] = np.abs(median_timestep - median_initial)

        #90th percentile
        high_initial = np.nanpercentile(np.stack(initial_conditions[layer]), 90, axis=0)
        high_timestep = np.nanpercentile(np.stack(conditions[layer]), 90, axis=0)
        absolute_diff_90th[layer] = np.abs(high_timestep - high_initial)

        #10th percentile
        low_initial = np.nanpercentile(np.stack(initial_conditions[layer]), 10, axis=0)
        low_timestep = np.nanpercentile(np.stack(conditions[layer]), 10, axis=0)
        absolute_diff_10th[layer] = np.abs(low_timestep - low_initial)
    
    return absolute_diff_median, absolute_diff_90th, absolute_diff_10th

#calculate differences for 25, 50, and 75 years
diff_median_25, diff_90th_25, diff_10th_25 = calculate_differences(initial_conditions, conditions_25_years, "25 years")
diff_median_50, diff_90th_50, diff_10th_50 = calculate_differences(initial_conditions, conditions_50_years, "50 years")
diff_median_75, diff_90th_75, diff_10th_75 = calculate_differences(initial_conditions, final_conditions, "75 years")

#customize plots
colors = ['white', 'lavenderblush', 'lightpink', 'salmon', 'indianred', 'maroon', 'black']
bounds = [0, 40, 70, 140, 200, 250, 350, 500]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

#read well locations
wells = pd.read_excel("../NEW_inputs/pumpingwell_locations.xlsx")
well_row = wells.iloc[:, 1].values
well_col = wells.iloc[:, 2].values
well_ids = wells.iloc[:, 3].values

#customize wells
def get_well_color(well_id):
    if well_id in [2718, 2723, 2727, 2732, 2759, 2761, 2762, 2799, 2803, 2804, 2805, 2806, 2940, 2943, 2953, 2963]:
        return 'mediumturquoise'
    else:
        return 'lightyellow'

def get_well_edge(well_id):
    if well_id in [2718, 2723, 2727, 2732, 2759, 2761, 2762, 2799, 2803, 2804, 2805, 2806, 2940, 2943, 2953, 2963]:
        return 'black'
    else:
        return 'lightyellow'

def get_well_size(well_id):
    if well_id in [2718, 2723, 2727, 2732, 2759, 2761, 2762, 2799, 2803, 2804, 2805, 2806, 2940, 2943, 2953, 2963]:
        return 77
    else:
        return 10

### plot for each layer and scenario ###
#FOR YEAR 75
for layer in diff_median_75:
    #median plot
    fig, ax = plt.subplots(figsize=(10, 8))
    abs_diff_masked = np.ma.masked_invalid(diff_median_75[layer])
    im = ax.imshow(abs_diff_masked, cmap=cmap, norm=norm, aspect='equal', alpha=0.9)
    plt.colorbar(im, ax=ax, label="Cl concentration (ppm)", extend='max')
    ax.set_facecolor('gainsboro')
    well_colors = [get_well_color(well_id) for well_id in well_ids]
    well_edge = [get_well_edge(well_id) for well_id in well_ids]
    well_size = [get_well_size(well_id) for well_id in well_ids]
    plt.scatter(well_col, well_row, color=well_colors, edgecolor=well_edge, s=well_size, marker='o')
    plt.title(f"Layer {layer + 1} - Final Change in Median Chloride")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"layer_{layer + 1}_final_diff_median.svg", format="svg")
    plt.close()

for layer in diff_90th_75:
    #90th percentile plot
    fig, ax = plt.subplots(figsize=(10, 8))
    abs_diff_masked = np.ma.masked_invalid(diff_90th_75[layer])
    im = ax.imshow(abs_diff_masked, cmap=cmap, norm=norm, aspect='equal', alpha=0.9)
    plt.colorbar(im, ax=ax, label="Cl concentration (ppm)", extend='max')
    ax.set_facecolor('gainsboro')
    plt.scatter(well_col, well_row, color=well_colors, edgecolor=well_edge, s=well_size, marker='o')
    plt.title(f"Layer {layer + 1} - Final Change in 90th Percentile Chloride")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"layer_{layer + 1}_final_diff_90th.svg", format="svg")
    plt.close()


#FOR YEAR 25
for layer in diff_median_25:
    #median plot
    fig, ax = plt.subplots(figsize=(10, 8))
    abs_diff_masked = np.ma.masked_invalid(diff_median_25[layer])
    im = ax.imshow(abs_diff_masked, cmap=cmap, norm=norm, aspect='equal', alpha=0.9)
    plt.colorbar(im, ax=ax, label="Cl concentration (ppm)", extend='max')
    ax.set_facecolor('gainsboro')
    well_colors = [get_well_color(well_id) for well_id in well_ids]
    well_edge = [get_well_edge(well_id) for well_id in well_ids]
    well_size = [get_well_size(well_id) for well_id in well_ids]
    plt.scatter(well_col, well_row, color=well_colors, edgecolor=well_edge, s=well_size, marker='o')
    plt.title(f"Layer {layer + 1} - Yr 25 Change in Median Chloride")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"layer_{layer + 1}_yr25_diff_median.svg", format="svg")
    plt.close()

for layer in diff_median_25:
    #90th percentile plot
    fig, ax = plt.subplots(figsize=(10, 8))
    abs_diff_masked = np.ma.masked_invalid(diff_median_25[layer])
    im = ax.imshow(abs_diff_masked, cmap=cmap, norm=norm, aspect='equal', alpha=0.9)
    plt.colorbar(im, ax=ax, label="Cl concentration (ppm)", extend='max')
    ax.set_facecolor('gainsboro')
    plt.scatter(well_col, well_row, color=well_colors, edgecolor=well_edge, s=well_size, marker='o')
    plt.title(f"Layer {layer + 1} - Yr 25Change in 90th Percentile Chloride")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"layer_{layer + 1}_yr25_diff_90th.svg", format="svg")
    plt.close()


#FOR YEAR 50
for layer in diff_median_50:
    #median plot
    fig, ax = plt.subplots(figsize=(10, 8))
    abs_diff_masked = np.ma.masked_invalid(diff_median_50[layer])
    im = ax.imshow(abs_diff_masked, cmap=cmap, norm=norm, aspect='equal', alpha=0.9)
    plt.colorbar(im, ax=ax, label="Cl concentration (ppm)", extend='max')
    ax.set_facecolor('gainsboro')
    well_colors = [get_well_color(well_id) for well_id in well_ids]
    well_edge = [get_well_edge(well_id) for well_id in well_ids]
    well_size = [get_well_size(well_id) for well_id in well_ids]
    plt.scatter(well_col, well_row, color=well_colors, edgecolor=well_edge, s=well_size, marker='o')
    plt.title(f"Layer {layer + 1} - Yr 50 Change in Median Chloride")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"layer_{layer + 1}_yr50_diff_median.svg", format="svg")
    plt.close()

for layer in diff_median_50:
    #90th percentile plot
    fig, ax= plt.subplots(figsize=(10, 8))
    abs_diff_masked = np.ma.masked_invalid(diff_median_50[layer])
    im = ax.imshow(abs_diff_masked, cmap=cmap, norm=norm, aspect='equal', alpha=0.9)
    plt.colorbar(im, ax=ax, label="Cl concentration (ppm)", extend='max')
    ax.set_facecolor('gainsboro')
    plt.scatter(well_col, well_row, color=well_colors, edgecolor=well_edge, s=well_size, marker='o')
    plt.title(f"Layer {layer + 1} - Yr 50 Change in 90th Percentile Chloride")
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"layer_{layer + 1}_yr50_diff_90th.svg", format="svg")
    plt.close()





#only print 16 wells

selected_well_ids = [2718, 2723, 2727, 2732, 2759, 2761, 2762, 2799, 2803, 2804, 2805, 2806, 2940, 2943, 2953, 2963]
mask = np.isin(well_ids, selected_well_ids)
filtered_well_row = well_row[mask]
filtered_well_col = well_col[mask]
filtered_well_ids = well_ids[mask]

#print plot aty layer 3 with only important 16 wells
fig, ax = plt.subplots(figsize=(10, 8))
abs_diff_masked = np.ma.masked_invalid(diff_median_75[2])
im = ax.imshow(abs_diff_masked, cmap=cmap, norm=norm, aspect='equal')
plt.colorbar(im, ax=ax, label="Cl concentration (ppm)", extend='max')
ax.set_facecolor('gainsboro')
plt.scatter(filtered_well_col, filtered_well_row, color='mediumturquoise', edgecolor='black', s=65, marker='o')
plt.title(f"Median Chloride")
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.savefig(f"wells_diff_median.svg", format="svg")
plt.close()
