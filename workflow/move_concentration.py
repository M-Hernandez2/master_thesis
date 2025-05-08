#Author: Madison Hernandez
#Purpose: go into each of the 500 folders of model runs, take output files, rename, add to folder on the cluster

import os
import shutil


og_folder_path = '../NEW_inputs/Final_Runs/'
new_folder_path = '../NEW_inputs/Final_Runs/CONCENTRATIONS/'

#loop through the subfolders in the 'final' folder
for subfolder in os.listdir(og_folder_path):
    subfolder_path = os.path.join(og_folder_path, subfolder)

    #check if it's a folder and matches the RunSLR#P# pattern
    if os.path.isdir(subfolder_path) and subfolder.startswith("RunSLR"):
        # Construct the new file name
        new_file_name = f"{subfolder}.OBS"
        file_to_move = os.path.join(subfolder_path, "MT3D001.UCN")

        #check if the file exists
        if os.path.exists(file_to_move):
            # Move and rename the file
            shutil.move(file_to_move, os.path.join(new_folder_path, new_file_name))

print("All files have been moved and renamed successfully.")
