#Author: Madison Hernandez, Rachel Housego, Alex Thames
#Purpose: code to take input data to create input files for many runs of seawat

import os
import shutil
import pandas as pd
import numpy as np

#### Read in boundary condition files ####
well_sims = pd.read_excel('..\NEW_inputs\well_irrigation_simulation_NEW.xlsx', sheet_name=None)
new_recharge = pd.read_excel('..\NEW_inputs\r_sequence.xlsx')
slr_scenarios = pd.read_excel('..\NEW_inputs\sealevelrise_model_inputs.xlsx',sheet_name=0)
#Path to reference model
base_model_path = '..\NEW_inputs\Base_Model'


#### SET UP LOOP OVER ALL SIMULATION CASES
for j in range(0, 1): #SLR scenarios   #Max range 5
  slr_vals = slr_scenarios.iloc[:,j]  #Max range 100 made smaller to test
  for k in range(1): #Precip + pumping scenarios

      #### CREATE CURRENT DIRECTORY ####

      #Make new folder to save input files for first run
      new_model_path = f'..\NEW_inputs\Final_Runs\RunSLR{(j+1)}P{(k+1)}'
      os.makedirs(new_model_path, exist_ok=True)
      ##

      #### COPY BASE MODEL FILES #####
      # Define source and destination folders
      src_folder = base_model_path
      dest_folder = new_model_path

      # Define the file extensions to exclude
      exclude_extensions = ('.WEL', '.RIV', '.CHD')

      # Ensure the destination folder exists
      if not os.path.exists(dest_folder):
          os.makedirs(dest_folder)

      # Walk through each file and directory in the source folder
      for root, dirs, files in os.walk(src_folder):
          # Determine the corresponding path in the destination folder
          relative_path = os.path.relpath(root, src_folder)
          dest_path = os.path.join(dest_folder, relative_path)

          # Ensure each subdirectory exists in the destination
          if not os.path.exists(dest_path):
              os.makedirs(dest_path)

          # Copy files that don't have the excluded extensions
          for file in files:
              if not any(file.endswith(ext) for ext in exclude_extensions) and "recharge" not in file:
                  src_file = os.path.join(root, file)
                  dest_file = os.path.join(dest_path, file)
                  shutil.copy2(src_file, dest_file)
      print('Base model files copied to new directory')

      #### EDIT RECHARGE FILES ####

      precip = new_recharge.iloc[:, k]  # CHANGE TO SIMULATION INDEX

      for i, val in enumerate(precip):
          # Define file paths for reading from the base model path and writing to the new model path
          file_path = os.path.join(base_model_path, f"EastDoverSWI.RCH.recharge{i + 1}")
          new_file_path = os.path.join(new_model_path, f"EastDoverSWI.RCH.recharge{i + 1}")

          # Convert val to the desired units (assuming mm to meters here)
          new_val = val / 1000  # Check if units conversion is correct

          # Check if the file exists before processing
          if os.path.exists(file_path):
              # Read the recharge text file for that stress period
              with open(file_path, 'r') as file:
                  file_content = file.readlines()

              # Process each line, replacing non-zero values with new_val
              modified_content = [
                  ' '.join([f"{new_val:.5f}" if float(value) != 0 else value for value in line.split()]) + "\n"
                  for line in file_content
              ]

              # Write the modified content back to the new file path
              with open(new_file_path, 'w') as file:
                  file.writelines(modified_content)
      print("Recharge files modified and saved successfully.")

      #### EDIT WELL FILE ####

      # read in the .WEL file
      with open(os.path.join(base_model_path, 'EastDoverSWI.WEL'), "r") as f:
          wel_data = f.readlines()

      # two lines are special (both start with 250), so specify them
      wel_data_special_250_163 = wel_data[0]
      wel_data_special_250_0 = wel_data[1]

      # make an empty dictionary to hold the important values from each line
      wel_dict, instance = {}, -2
      for l in range(len(wel_data)):
          line = wel_data[l]
          # -- eliminate newline characters, split the line into a new element on each occurrence of a space character
          split_line = line.replace("\n", "").split(" ")
          # -- get rid of all the elements with empty characters
          stripped_line = [el for el in split_line if el != ""]

          # if the number of elements in the line == 2, move on
          if len(stripped_line) == 2:
              instance += 1
              continue

          # add it to the dictionary
          wel_dict[l] = [instance, int(stripped_line[0]),
                         int(stripped_line[1]),
                         int(stripped_line[2]),
                         float(stripped_line[3]),
                         int(stripped_line[4])]

      # convert dictionary to a dataframe, with .WEL line number as index
      # -- not sure what the columns are so I just made titles up
      wel_DF = pd.DataFrame().from_dict(wel_dict, orient="index",
                                        columns=["INSTANCE", "LAYER", "A", "B", "AMOUNT", "ID"])

      # where are the instance start and end lines?
      inst_startlines = [l for l in list(wel_DF.index) if wel_DF.at[l, "ID"] == 2718][::3]
      inst_endlines = [l - 2 for l in inst_startlines[1:]]
      inst_endlines.append(wel_DF.index[-1])

      # open a new .WEL file and start to write to it
      with open(os.path.join(new_model_path, 'EastDoverSWI.WEL'), "w") as simFile:
          # first line of the file is unique
          simFile.write(wel_data_special_250_163)

          # -- for each instance...
          instances = sorted(set(wel_DF["INSTANCE"].values))
          # !!!!!!!!!!!!!!!!! NOTE !!!!!!!!!!!!!!!!!!!!!!
          # in your .xlsx file, you have 75 instances for each well ID
          # in your  .WEL file, you have 76 instances for each well ID
          # -- I think these should be the same, so I'm dropping the 76th instance when writing to the .WEL file
          # -- add another instance to our .xlsx file, then delete the [:-1] below
          # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          for inst in instances[:-1]:  # Do final check on this Rachel had to make -2 to get to run
              # write the start of the instance
              simFile.write(wel_data_special_250_0)

              # get the start and end lines of the instance
              l_start, l_end = inst_startlines[inst], inst_endlines[inst]

              # -- for these lines...
              for l in range(l_start, l_end + 1):
                  # start the simulated .WEL line
                  sim_line = ""

                  # setting up logical indexing on the instance, ID
                  line_layer, line_A, line_B = wel_DF.at[l, "LAYER"], wel_DF.at[l, "A"], wel_DF.at[l, "B"]
                  line_inst, line_ID = wel_DF.at[l, "INSTANCE"], wel_DF.at[l, "ID"]
                  line_inst_idx, line_ID_idx = wel_DF["INSTANCE"] == line_inst, wel_DF["ID"] == line_ID

                  # find the layers/pumping for this instance/ID combo
                  wel_layers = wel_DF.loc[line_ID_idx & line_inst_idx, "LAYER"].values
                  wel_amounts = wel_DF.loc[line_ID_idx & line_inst_idx, "AMOUNT"].values
                  # sum, get fractions
                  wel_total = sum(wel_amounts)
                  wel_fracs = wel_amounts / wel_total if abs(wel_total) > 0. else np.array([1.])
                  # make sure we can find a simulation for this ID...if not, don't change the line
                  if str(line_ID) in list(well_sims.keys()):
                      # divide up fractions; get this line's specific pumping fraction
                      frac_pumpings = well_sims[str(line_ID)][k].values[inst] * wel_fracs  # CHANGE LINE TO SIMULATION INDEX
                      frac_pumping = [val for v, val in enumerate(frac_pumpings) if wel_layers[v] == line_layer][0]
                  else:
                      frac_pumping = wel_DF.at[l, "AMOUNT"]

                  # write the simulated well line
                  # -- layer, A, B: right-justified to 10
                  sim_line += str(line_layer).rjust(10, " ")
                  sim_line += str(line_A).rjust(10, " ")
                  sim_line += str(line_B).rjust(10, " ")
                  # -- amount: only the first 7 characters (5 digits, ., and -), then right-justified to 11
                  sim_line += str(frac_pumping)[:7].rjust(11, " ")
                  # -- ID: no justification
                  sim_line += " " + str(line_ID) + "\n"
                  # -- write!
                  simFile.write(sim_line)
      print("WEL file modified and saved successfully.")

      #### EDIT THE RIV FILE ####

      # Load the MODFLOW river file, retaining the header and reading in the data
      input_file_path = os.path.join(base_model_path, 'EastDoverSWI.RIV')  # Path to base model river file
      output_file_path = os.path.join(new_model_path, 'EastDoverSWI.RIV')  # Output file path

      # Initialize list to hold modified content
      modified_content = []

      # Initialize list to hold modified content
      modified_content = []

      # Dictionary to track the updated fourth column values by row position within each stress period
      previous_stress_period_values = {}

      # Open the input file
      with open(input_file_path, 'r') as file:
          # Keep track of the stress period and current increment value
          stress_period_index = -1  # This will start from 0 on the first "1761" line
          row_position = 0  # Track row position within a stress period

          # Process each line in the file
          for line_num, line in enumerate(file):
              values = line.strip().split()

              # Keep the header row exactly as it is, without modification or indentation
              if line_num == 0:
                  modified_content.append(line)
                  continue

              # Check if this line marks a new stress period
              if len(values) == 1 and values[0] == "1761":
                  # Move to the next stress period and update the increment value
                  stress_period_index += 1
                  if stress_period_index == 0:
                      increment_value = slr_vals[stress_period_index]
                      # Get increment value for the current stress period
                  elif stress_period_index >0 and stress_period_index <75:
                      increment_value = slr_vals[stress_period_index] - slr_vals[stress_period_index-1]

                  # Append this separator line as-is
                  modified_content.append("1761\n")

                  # Reset the row position and the `previous_stress_period_values` for the new stress period
                  row_position = 0
                  continue

              # Modify the fourth column if it's a data row and stress period is within range
              elif len(values) >= 7:
                  try:
                      # For the first stress period, use the original value as the base
                      if stress_period_index == 0:
                          new_value = float(values[3]) + increment_value if float(values[3]) != 0 else float(values[3])
                      else:
                          # For subsequent stress periods, add the increment to the previous stress period's value at the same row position
                          previous_value = previous_stress_period_values.get(row_position, float(values[3]))
                          new_value = previous_value + increment_value if previous_value != 0 else previous_value

                      # Store the updated value for the current row position in the dictionary
                      previous_stress_period_values[row_position] = new_value

                      # Update the fourth and sixth columns, formatting the sixth column to five decimal places
                      values[3] = f"{new_value:.5f}"
                      values[5] = f"{float(values[5]):.5f}"

                      # Reformat the modified line to maintain consistent spacing, including space between column 6 and 7
                      formatted_line = "{:<6}  {:<6}  {:<6}  {:<12}  {:<12}  {:<12}    {:<12}".format(*values)
                      modified_content.append(formatted_line + "\n")

                      # Increment the row position within the stress period
                      row_position += 1

                  except (ValueError, IndexError) as e:
                      # Handle conversion or index error
                      print(f"Skipping line due to error: {line}, Error: {e}")
                      continue

              else:
                  # If the line format doesn't match expectations, keep it as-is
                  modified_content.append(line)

      # Write the modified content back to a new file without indenting the header row
      with open(output_file_path, 'w') as output_file:
          output_file.writelines(modified_content)

      print("RIV file modified and saved successfully.")

      #### EDIT CHD FILES ####
      time_series = slr_vals

      # File paths
      input_file_path = os.path.join(base_model_path, 'EastDoverSWI.CHD')
      output_file_path = os.path.join(new_model_path, 'EastDoverSWI.CHD')

      # Initialize output content
      modified_content = []
      time_series_index = -1  # Track the current position in the time series

      # Read the file line by line
      with open(input_file_path, 'r') as file:
          for line in file:
              values = line.strip().split()

              # Check if this line indicates a new stress period
              if len(values) == 2 and values[0] == "2256" and values[1] == "163":
                  # Start a new stress period, append the separator line as is
                  modified_content.append(line)
                  time_series_index += 1
                  continue

              # If line has enough columns (indicating it’s a data line), modify start and end heads
              elif len(values) >= 7:
                  try:
                      if time_series_index == 0:
                          start_head = 0;
                          end_head = time_series[time_series_index]
                      # Use current time series value for the start and end head
                      elif time_series_index >0 and time_series_index <75:
                          start_head = time_series[time_series_index - 1]
                          end_head = time_series[
                              time_series_index]  # Assuming start and end heads are the same for simplicity

                      # Update start and end head values in the line
                      values[3] = f"{start_head:.5f}"
                      values[4] = f"{end_head:.5f}"

                      # Format line to match original layout and spacing
                      formatted_line = "{:<6} {:<6}  {:<6}  {:<12}  {:<12}  {:<6}  {:<12}".format(*values)
                      modified_content.append(formatted_line + "\n")

                  except IndexError:
                      print("Warning: Time series does not have enough values for all stress periods.")
                      break

              else:
                  # Keep any other lines unmodified
                  modified_content.append(line)

      # Write the modified content back to a new file
      with open(output_file_path, 'w') as output_file:
          output_file.writelines(modified_content)

      print("CHD file modified and saved successfully.")
