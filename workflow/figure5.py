
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import precipitation and temperature data as well as wells (this is the NEW CLIMATE DATA)
temp = pd.read_excel('C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\\NEW_all_P_T_proj.xlsx', sheet_name='t_timeseries')
precip = pd.read_excel('C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\\NEW_all_P_T_proj.xlsx', sheet_name='p_timeseries')
wells = pd.read_excel('C:/Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SEAWAT_model_inputs\wellsareas_split.xlsx')

#convert precipitation to recharge by getting 13% of precip = recharge
rech = precip * 0.13
print(rech)

# using the Thornthwaite method for calculating Evapotranspiration, adjusted to annual scale, not monthly
def apply_et(t):
    I = 12* ((t/5)**1.514)
    alpha = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (0.01792 * I) + (0.49239)

    return (16 * (10 * t / I)**alpha) * 12    #*12 to get annual

eto = apply_et(temp)
print(eto)

#cumulative crop coeficients for all plants in Dover with well, cumulative across all growth stages, Kc
corn_kc = 1.2
soy_kc = 1.65
potato_kc = 1.90
wwheat_kc = 2.175
hay_kc = 2.52

#STEP 1: calc the evapotranspiration of each crop type, ETc
def crop_et(et, crop):
    return et * crop

et_corn = pd.DataFrame(crop_et(eto, corn_kc))
et_soy = pd.DataFrame(crop_et(eto, soy_kc))
et_potato = pd.DataFrame(crop_et(eto, potato_kc))
et_wwheat = pd.DataFrame(crop_et(eto, wwheat_kc))
et_hay = pd.DataFrame(crop_et(eto, hay_kc))


#STEP 2: get net irrigation requierement by subtract ETc from recharge, NIR
nir_corn = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
nir_soy = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
nir_potato = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
nir_wwheat = pd.DataFrame(np.nan, index=rech.index, columns=rech.columns)
for row in range(rech.shape[0]):
    for col in range(rech.shape[1]):
        nir_corn.iloc[row, col] = (et_corn.iloc[row, col] - rech.iloc[row, col])
        nir_soy.iloc[row, col] = (et_soy.iloc[row, col] - rech.iloc[row, col])
        nir_potato.iloc[row, col] = (et_potato.iloc[row, col] - rech.iloc[row, col])
        nir_wwheat.iloc[row, col] = (et_wwheat.iloc[row, col] - rech.iloc[row, col])

print('corn: ', nir_corn)
print('potato: ', nir_potato)

#creat demand for double crops whioch are all soy and  winter wheat
nir_dbl = (nir_soy + nir_wwheat) /2
print('dbl: ', nir_dbl)

#STEP 3: calc gross irrigation requirement, NIR/efficeincy factor, GIR
eff = 0.13   #efficiency factor, 13% of precip = rech

#STEP 4: calc volume of water to be pumped, NIR*area, vol
#make a dictionary to  hold the simulations for each well
well_dict = {}

for i, row in wells.iterrows():
    key = row['well_id']    #set keys to dict as well ids

    results = [[] for _ in range(nir_corn.shape[1])]
    for i in range(len(nir_corn)):
            for j, col in enumerate(nir_corn.columns):

                if row['crop'] == 'corn' and row['crop2'] == 'corn':
                    d = (nir_corn[col].iloc[i] * row['area'] + nir_corn[col].iloc[i] * row['area2']).tolist()
                    results[j].append(d / 1000 * 0.26)
                elif row['crop'] == 'dbl' and row['crop2'] == 'wwheat':
                    d = (nir_dbl[col].iloc[i] * row['area'] + nir_wwheat[col].iloc[i] + row['area2']).tolist()
                    results[j].append(d / 1000)
                elif row['crop'] == 'soy' and row['crop2'] == 'dbl':
                    d = (nir_soy[col].iloc[i] * row['area'] + nir_dbl[col].iloc[i] * row['area2']).tolist()
                    results[j].append(d / 1000)
                elif row['crop'] == 'corn' and row['crop2'] == 'dbl':
                    d = (nir_corn[col].iloc[i] * row['area'] + nir_dbl[col].iloc[i] * row['area2']).tolist()
                    results[j].append(d / 1000)
                elif row['crop'] == 'soy' and row['crop2'] == 'corn':
                    d = (nir_soy[col].iloc[i] * row['area'] + nir_corn[col].iloc[i] * row['area2']).tolist()
                    results[j].append(d / 1000 * 0.38)

                elif row['crop'] == 'corn':
                    d = (nir_corn[col].iloc[i] * row['area']).tolist()
                    results[j].append(d / 1000 * 0.26)
                elif row['crop'] == 'soy':
                    d = (nir_soy[col].iloc[i] * row['area']).tolist()
                    results[j].append(d / 1000 * 0.38)
                elif row['crop'] == 'pot':
                    d = (nir_potato[col].iloc[i] * row['area']).tolist()
                    results[j].append(d / 1000 * 0.36)
                elif row['crop'] == 'dbl':
                    d = (nir_dbl[col].iloc[i] * row['area']).tolist()
                    results[j].append(d / 1000 * 0.79)

    well_dict[key] = results

well_dict = {key: list(map(list, zip(*val))) for key, val in well_dict.items()}

#STEP 5: convert volume to pumping rate, vol/T
#divide by 365 for days, make negative because its water being removed from the system
for key in well_dict:
    well_dict[key] = [[round(val / 365 * -1) for val in sublist] for sublist in well_dict[key]]


#save the dictionary to an excel file with each well being its own sheet
with pd.ExcelWriter('C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SEAWAT_model_inputs\well_irrigation_simulation_NEW.xlsx') as writer:
    for key, val in well_dict.items():
        df = pd.DataFrame(val)
        df.to_excel(writer, sheet_name=str(key), index=False)


#use historic precip and temp values to plot on graphs like projected and see if in range of Bulletin 22 from DGS
hist = pd.read_excel('C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\histPT_irrigation.xlsx')
hist_temp = hist['t_avg']          #degrees C
hist_rech = hist['rech']           #mm/yr

avgT = sum(hist_temp) / len(hist_temp)
avgR = sum(hist_rech) / len(hist_rech)

hist_eto = apply_et(avgT)

#STEP 1: calc the evapotranspiration of each crop type, ETc
h_et_corn = crop_et(hist_eto, corn_kc)
h_et_soy = crop_et(hist_eto, soy_kc)
h_et_potato = crop_et(hist_eto, potato_kc)
h_et_wwheat = crop_et(hist_eto, wwheat_kc)
h_et_hay = crop_et(hist_eto, hay_kc)

print('avg r', avgR)
print('et corn ',h_et_corn)
print('et potato ',h_et_potato)

#STEP 2: get net irrigation requierement by subtract ETc from recharge, NIR
nir_corn_h = h_et_corn - avgR
nir_soy_h = h_et_soy - avgR
nir_potato_h = h_et_potato - avgR
nir_wwheat_h = h_et_wwheat - avgR
nir_dbl_h = (nir_soy_h + nir_wwheat_h) /2

print('nir corn ',nir_corn_h)
print('nir dbl ',nir_dbl_h)

well_lst = wells.set_index(wells.columns[0])
results = []

for i, row in wells.iterrows():
    key = row['well_id']
    print(key)

    if row['crop'] == 'corn' and row['crop2'] == 'corn':
        d = (nir_corn_h * row['area'] + nir_corn_h * row['area2'])
        results.append(d / 1000 * 0.26)
    elif row['crop'] == 'dbl' and row['crop2'] == 'wwheat':
        d = (nir_dbl_h * row['area'] + nir_wwheat_h + row['area2'])
        results.append(d / 1000)
    elif row['crop'] == 'soy' and row['crop2'] == 'dbl':
        d = (nir_soy_h * row['area'] + nir_dbl_h * row['area2'])
        results.append(d / 1000)
    elif row['crop'] == 'corn' and row['crop2'] == 'dbl':
        d = (nir_corn_h * row['area'] + nir_dbl_h * row['area2'])
        results.append(d / 1000)
    elif row['crop'] == 'soy' and row['crop2'] == 'corn':
        d = (nir_soy_h * row['area'] + nir_corn_h * row['area2'])
        results.append(d / 1000 * 0.38)

    elif row['crop'] == 'corn':
        d = (nir_corn_h * row['area'])
        results.append(d / 1000 * 0.26)
    elif row['crop'] == 'soy':
        d = (nir_soy_h * row['area'])
        results.append(d / 1000 * 0.38)
    elif row['crop'] == 'pot':
        d = (nir_potato_h * row['area'])
        results.append(d / 1000 * 0.36)
    elif row['crop'] == 'dbl':
        d = (nir_dbl_h* row['area'])
        results.append(d / 1000 * 0.79)

#STEP 5: convert volume to pumping rate, vol/T
#divide by 365 for days
daily_pump = []
for i in results:
    r = i / 365
    daily_pump.append(r)

hist_df = pd.DataFrame(data=daily_pump, index=wells['well_id'], columns =['pumping'])
print(hist_df)

#save to an excel file
hist_df.to_excel('historic_thornthwaite_pumping.xlsx')



#use the excel sheets for plotting
wells = ('C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SEAWAT_model_inputs\well_irrigation_simulation_NEW.xlsx')
sheets = pd.ExcelFile(wells).sheet_names

#import farm areas associated with each well to order the figure based on farm size
irrig = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SEAWAT_model_inputs\wellsareas.xlsx",sheet_name='area_sort')
irrig = pd.DataFrame(irrig)
well_order = irrig['well_id']
crops = irrig['crop']

#reorder hist based on farm size
hist_est = hist_df.reindex(irrig['well_id']).reset_index()

crops = irrig['crop'].tolist()
calc = hist_est.iloc[:,1].tolist()
calc = [-abs(x) for x in calc]
val = irrig['og_pump'].tolist()

#plot with the average estimated pumping for each of the 69 irrigation wells in the study area
fig, axs = plt.subplots(3, 23, figsize=(12,14), sharey=True, constrained_layout=True)
axs = axs.flatten()

for i, well_name in enumerate(well_order):
    df = pd.read_excel(wells, sheet_name=str(well_name))
    df_pump = irrig.loc[irrig['well_id'] == well_name, 'og_pump'].values[0]
    col_mean = df.mean()
    sheet = sheets[i]

    axs[i].hlines(col_mean, xmin=0, xmax=1, color='navy', label=sheet, alpha=0.7)
    axs[i].axhline(y=calc[i], color='mediumturquoise', label='calculated',linewidth=3)
    axs[i].axhline(y=val[i], color='maroon', label='original model', linewidth=3, alpha=0.7)

    axs[i].set_title(well_order[i], fontsize=6, loc='left')
    axs[i].set_xticks([])
    axs[i].set_ylim(bottom=-3500, top=0)

for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])
fig.text(0.5, 0.04, 'pumping rate (m3/day)')
plt.tight_layout()
plt.legend()
plt.show()
