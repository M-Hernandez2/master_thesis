#Author: Madison Hernandez
#Purpose: bias correct the precip and temp data, then sample out, then create stochastic sequences to be used in SEAWAT

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import chaospy as cp

#retry from orig code but cleaner wand with new data
### HISTORICAL OBSERVED DATA ###
n_P_hist = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx", sheet_name='noaa_P_hist')
n_T_hist = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='noaa_T_hist')

hist_year = n_P_hist['YEAR']
noaa_P_hist = n_P_hist['observed_p']
noaa_T_hist = n_T_hist['observed_t']

#gets means and standard deviations of hist observed
noaa_p_mean = np.mean(noaa_P_hist)
noaa_t_mean = np.mean(noaa_T_hist)
noaa_p_std = np.std(noaa_P_hist)
noaa_t_std = np.std(noaa_T_hist)

### HISTORICAL MODEL DATA ###
mod_P_hist = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='P_historical')
mod_T_hist = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='T_historical')

#data frame with means and stds for precip and temp historical models
#get name of each model
# Convert the lists to DataFrames first
hist_p_stats = pd.DataFrame([{'Model': col,
                            'Mean': mod_P_hist[col].mean(),
                            'Std': mod_P_hist[col].std()}
                           for col in mod_P_hist.columns[1:]])

hist_t_stats = pd.DataFrame([{'Model': col,
                            'Mean': mod_T_hist[col].mean(),
                            'Std': mod_T_hist[col].std()}
                           for col in mod_T_hist.columns[1:]])

# Now you can add columns to the DataFrames
hist_p_stats['Scale'] = noaa_p_std / hist_p_stats['Std']
hist_p_stats['Shift'] = noaa_p_mean - (hist_p_stats['Scale'] * hist_p_stats['Mean'])

# Apply the correction to the historical models
hist_p_aligned = mod_P_hist.copy()
for idx, row in hist_p_stats.iterrows():
    col = row['Model']
    scale = row['Scale']
    shift = row['Shift']
    hist_p_aligned[col] = mod_P_hist[col] * scale + shift

# Do the same for temperature
hist_t_stats['Scale'] = noaa_t_std / hist_t_stats['Std']
hist_t_stats['Shift'] = noaa_t_mean - (hist_t_stats['Scale'] * hist_t_stats['Mean'])

hist_t_aligned = mod_T_hist.copy()
for idx, row in hist_t_stats.iterrows():
    col = row['Model']
    scale = row['Scale']
    shift = row['Shift']
    hist_t_aligned[col] = mod_T_hist[col] * scale + shift

### APPLY BIAS CORRECTION TO PROJECTED MODELS ###
#precip projections
ssp26_p = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='Pssp2.6')
ssp45_p = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='Pssp4.5')
ssp70_p = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='Pssp7.0')
ssp85_p = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='Pssp8.5')

#temp projections
ssp26_t = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='Tssp2.6')
ssp45_t = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='Tssp4.5')
ssp70_t = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='Tssp7.0')
ssp85_t = pd.read_excel("C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\DoverClimate_Data.xlsx",sheet_name='Tssp8.5')

def align_proj(proj_df, stats_df):
    aligned_proj = proj_df.copy()

    for i, row in stats_df.iterrows():
        model = row['Model']
        if model in proj_df.columns:
            scale= row['Scale']
            shift = row['Shift']

            #check for missing data
            mask = proj_df[model].notna()
            aligned_proj.loc[mask, model] = proj_df.loc[mask, model] * scale + shift

    return aligned_proj

#apply correction to precip projections
ssp26_p_aligned = align_proj(ssp26_p, hist_p_stats)
ssp45_p_aligned = align_proj(ssp45_p, hist_p_stats)
ssp70_p_aligned = align_proj(ssp70_p, hist_p_stats)
ssp85_p_aligned = align_proj(ssp85_p, hist_p_stats)

#dictionary of precip scenarios
scenarios = {
    'SSP2.6': ssp26_p_aligned,
    'SSP4.5': ssp45_p_aligned,
    'SSP7.0': ssp70_p_aligned,
    'SSP8.5': ssp85_p_aligned
}

#combo them
corr_p = pd.concat(
    {scen: df.set_index('YEAR') for scen, df in scenarios.items()},
    axis=0, names=['Scenario', 'YEAR']
).reset_index()

#apply correction to temp projections
ssp26_t_aligned = align_proj(ssp26_t, hist_t_stats)
ssp45_t_aligned = align_proj(ssp45_t, hist_t_stats)
ssp70_t_aligned = align_proj(ssp70_t, hist_t_stats)
ssp85_t_aligned = align_proj(ssp85_t, hist_t_stats)

scenarios = {
    'SSP2.6': ssp26_t_aligned,
    'SSP4.5': ssp45_t_aligned,
    'SSP7.0': ssp70_t_aligned,
    'SSP8.5': ssp85_t_aligned
}

corr_t = pd.concat(
    {scen: df.set_index('YEAR') for scen, df in scenarios.items()},
    axis=0, names=['Scenario', 'YEAR']
).reset_index()

print(corr_t)


### VISUALIZATION OF BIAS CORRECTIONS ###

#historic precip distributions
plt.figure(figsize=(12, 8))
for model in mod_P_hist.columns[1:]:
    data = mod_P_hist[model].dropna()
    sns.kdeplot(data, color='navy', linewidth=2)

#calculate and plot average model distribution
all_model_data = np.concatenate([mod_P_hist[col].dropna() for col in mod_P_hist.columns[1:]])
avg_mu, avg_sigma = np.mean(all_model_data), np.std(all_model_data)
x = np.linspace(np.min(all_model_data)*0.8, np.max(all_model_data)*1.2, 500)
plt.plot(x, norm.pdf(x, avg_mu, avg_sigma),
         color='maroon', linewidth=4)

plt.xlim(500, 2000)
plt.ylim(0.0, 0.004)
plt.title('Precipitation Historic Distributions (Pre-Correction)', fontsize=14)
plt.xlabel('Precipitation (mm/yr)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.tight_layout()
plt.show()

#projected precip distributions with noaa hist observed
plt.figure(figsize=(12, 8))
sns.kdeplot(noaa_P_hist, color='maroon', linewidth=4, zorder=100)

#plot each bias-corrected model's distribution
for model in hist_p_aligned.columns[1:]:
    data = hist_p_aligned[model].dropna()
    sns.kdeplot(data, color='navy', linewidth=2)

#plot average of bias-corrected models
all_corrected_data = np.concatenate([hist_p_aligned[col].dropna() for col in hist_p_aligned.columns[1:]])
avg_mu_corrected, avg_sigma_corrected = np.mean(all_corrected_data), np.std(all_corrected_data)
x = np.linspace(min(noaa_P_hist.min(), all_corrected_data.min())*0.8,
                max(noaa_P_hist.max(), all_corrected_data.max())*1.2, 500)

plt.xlim(500, 2000)
plt.ylim(0.0, 0.004)
plt.title('Precipitation Distributions: Observed vs Bias-Corrected Models', fontsize=14)
plt.xlabel('Precipitation (mm/yr)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.tight_layout()
plt.show()


#historic temperature distributions
plt.figure(figsize=(12, 8))
for model in mod_T_hist.columns[1:]:
    data = mod_T_hist[model].dropna()
    sns.kdeplot(data, color='navy', linewidth=2)

#calculate and plot average model distribution
all_model_data = np.concatenate([mod_T_hist[col].dropna() for col in mod_T_hist.columns[1:]])
avg_mu, avg_sigma = np.mean(all_model_data), np.std(all_model_data)
x = np.linspace(np.min(all_model_data)*0.8, np.max(all_model_data)*1.2, 500)
plt.plot(x, norm.pdf(x, avg_mu, avg_sigma),
         color='maroon', linewidth=4)

plt.xlim(10, 20)
plt.ylim(0.0, 0.8)
plt.title('Temperature Historic Distributions (Pre-Correction)', fontsize=14)
plt.xlabel('Temperature (C)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.tight_layout()
plt.show()

#projected precip distributions with noaa hist observed
plt.figure(figsize=(12, 8))
sns.kdeplot(noaa_T_hist, color='maroon', linewidth=4, zorder=100)

#plot each bias-corrected model's distribution
for model in hist_t_aligned.columns[1:]:
    data = hist_t_aligned[model].dropna()
    sns.kdeplot(data, color='navy', linewidth=2)

#plot average of bias-corrected models
all_corrected_data = np.concatenate([hist_t_aligned[col].dropna() for col in hist_t_aligned.columns[1:]])
avg_mu_corrected, avg_sigma_corrected = np.mean(all_corrected_data), np.std(all_corrected_data)
x = np.linspace(min(noaa_T_hist.min(), all_corrected_data.min())*0.8,
                max(noaa_T_hist.max(), all_corrected_data.max())*1.2, 500)

plt.xlim(10, 20)
plt.ylim(0.0, 0.8)
plt.title('Precipitation Distributions: Observed vs Bias-Corrected Models', fontsize=14)
plt.xlabel('Temperature (C)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.tight_layout()
plt.show()



### SAMPLING USING THE DELTA METHOD ###
#get absolute max and mins for the mean and standard deviations of precip
#means
p_projected_means = {
    'SSP2.6': {},
    'SSP4.5': {},
    'SSP7.0': {},
    'SSP8.5': {}
}

#calc means for each scenario
for scenario, df in zip(['SSP2.6', 'SSP4.5', 'SSP7.0', 'SSP8.5'],
                       [ssp26_p_aligned, ssp45_p_aligned, ssp70_p_aligned, ssp85_p_aligned]):
    for model in df.columns[1:]:
        p_projected_means[scenario][model] = df[model].mean()

p_means_df = pd.DataFrame(p_projected_means)

print("Aligned Projected Means (mm/yr):")
print(p_means_df.round(1))

#get global max and min
all_scenarios = pd.concat([
    ssp26_p_aligned.set_index('YEAR').add_prefix('SSP2.6_'),
    ssp45_p_aligned.set_index('YEAR').add_prefix('SSP4.5_'),
    ssp70_p_aligned.set_index('YEAR').add_prefix('SSP7.0_'),
    ssp85_p_aligned.set_index('YEAR').add_prefix('SSP8.5_')
], axis=1)

p_means = all_scenarios.mean()

p_mean_max = p_means.max()
p_mean_min = p_means.min()

#STDs
p_projected_std = {
    'SSP2.6': {},
    'SSP4.5': {},
    'SSP7.0': {},
    'SSP8.5': {}
}

#calc means for each scenario
for scenario, df in zip(['SSP2.6', 'SSP4.5', 'SSP7.0', 'SSP8.5'],
                       [ssp26_p_aligned, ssp45_p_aligned, ssp70_p_aligned, ssp85_p_aligned]):
    for model in df.columns[1:]:
        p_projected_std[scenario][model] = df[model].std()

p_std_df = pd.DataFrame(p_projected_std)

print("Aligned Projected Means (mm/yr):")

#get global max and min
all_scenarios = pd.concat([
    ssp26_p_aligned.set_index('YEAR').add_prefix('SSP2.6_'),
    ssp45_p_aligned.set_index('YEAR').add_prefix('SSP4.5_'),
    ssp70_p_aligned.set_index('YEAR').add_prefix('SSP7.0_'),
    ssp85_p_aligned.set_index('YEAR').add_prefix('SSP8.5_')
], axis=1)

p_std = all_scenarios.std()

p_std_max = p_std.max()
p_std_min = p_std.min()

print(p_std_min, 'and', p_std_max)
print(p_mean_min, 'and', p_mean_max)


#get absolute max and mins for the mean and standard deviations of temp
#means
t_projected_means = {
    'SSP2.6': {},
    'SSP4.5': {},
    'SSP7.0': {},
    'SSP8.5': {}
}

#calc means for each scenario
for scenario, df in zip(['SSP2.6', 'SSP4.5', 'SSP7.0', 'SSP8.5'],
                       [ssp26_t_aligned, ssp45_t_aligned, ssp70_t_aligned, ssp85_t_aligned]):
    for model in df.columns[1:]:
        t_projected_means[scenario][model] = df[model].mean()

t_means_df = pd.DataFrame(t_projected_means)

print("Aligned Projected Means (C):")
print(t_means_df.round(1))

#get global max and min
t_all_scenarios = pd.concat([
    ssp26_t_aligned.set_index('YEAR').add_prefix('SSP2.6_'),
    ssp45_t_aligned.set_index('YEAR').add_prefix('SSP4.5_'),
    ssp70_t_aligned.set_index('YEAR').add_prefix('SSP7.0_'),
    ssp85_t_aligned.set_index('YEAR').add_prefix('SSP8.5_')
], axis=1)

t_means = t_all_scenarios.mean()

t_mean_max = t_means.max()
t_mean_min = t_means.min()

#standard deviations
t_projected_std = {
    'SSP2.6': {},
    'SSP4.5': {},
    'SSP7.0': {},
    'SSP8.5': {}
}

#calc means for each scenario
for scenario, df in zip(['SSP2.6', 'SSP4.5', 'SSP7.0', 'SSP8.5'],
                       [ssp26_t_aligned, ssp45_t_aligned, ssp70_t_aligned, ssp85_t_aligned]):
    for model in df.columns[1:]:
        t_projected_std[scenario][model] = df[model].std()

t_std_df = pd.DataFrame(t_projected_std)

print("Aligned Projected STDs (C):")

#get global max and min
t_all_scenarios = pd.concat([
    ssp26_t_aligned.set_index('YEAR').add_prefix('SSP2.6_'),
    ssp45_t_aligned.set_index('YEAR').add_prefix('SSP4.5_'),
    ssp70_t_aligned.set_index('YEAR').add_prefix('SSP7.0_'),
    ssp85_t_aligned.set_index('YEAR').add_prefix('SSP8.5_')
], axis=1)

t_std = t_all_scenarios.std()
t_std_max = t_std.max()
t_std_min = t_std.min()

print(t_std_min, 'and', t_std_max)
print(t_mean_min, 'and', t_mean_max)


#SAMPLING SPACE OUT OF MEANS AND STANDARD DEVIATIONS FOR P AND T TO USE FOR THEN MAKING TIME SERIES
#four dimensional sampling space with 200 samples
t_mean_range = [t_mean_min, t_mean_max]
t_std_range = [t_std_min, t_std_max]
p_mean_range = [p_mean_min, p_mean_max]
p_std_range = [p_std_min, p_std_max]

distribution = cp.J(cp.Uniform(p_mean_range[0], p_mean_range[1]),
                    cp.Uniform(p_std_range[0], p_std_range[1]),
                    cp.Uniform(t_mean_range[0], t_mean_range[1]),
                    cp.Uniform(t_std_range[0], t_std_range[1]))

#rule = 'L' indicated using latin-hypercube sampling method
samples = distribution.sample(100, rule='L')

#put it all into a dataframe and save it to excel file
sample_space_df = pd.DataFrame(samples.T, columns=['p mean', 'p standard deviation', 't mean', 't standard deviation'])

#to add the direct average means and stds from each of the CMIP6 models (post bias correction) to the pairplot
additional_data = pd.DataFrame({
    'p mean': p_means,
    'p standard deviation': p_std,
    't mean': t_means,
    't standard deviation': t_std})

additional_data['source'] = 'Original Data'
sample_space_df['source'] = 'Sampled Data'
combined_df = pd.concat([sample_space_df, additional_data], ignore_index=True)


#observed historic p mean, std, t mean, std from NOAA weather gauge
hist = [1111.2, 213, 13.55, 0.6697]

#create plot of 4d sampling space
g = sns.pairplot(combined_df, hue='source', corner=True, palette={'Sampled Data': 'maroon', 'Original Data': 'salmon' })

#add historic points
for i, row in enumerate(g.axes):
    for j, ax in enumerate(row):
        if ax is not None:
            if i == j:
                ax.axvline(hist[i], color='navy', linestyle='--', label='Historic Value')
            else:
                ax.scatter(hist[j], hist[i], color='navy', s=70, label='Historic Value')
            if i == 0 and j == 0:
                ax.legend()

plt.show()

print('SAMPLE SPACE', sample_space_df)
sample_space_df.to_excel('C:\\Users\mjh7517\OneDrive - The Pennsylvania State University\Downloads\SEAWAT_model_inputs\PT_mean_std.xlsx', index=False)

#CREATE SYNTHETIC TIME SERIES FOR PRECIPITATION AND TEMP USING SAMPLE SPACE BASED ON MEANS AND STDS

n = 75              #num of years to create time series for

dates = pd.date_range(start='2025', periods=n, freq='YE')


#RANDOM STOCHASTIC SEQUENCE FOR SEAWAT RUNS
#precipitation
p_time_series = pd.DataFrame(index=dates)
rech_timeseries = pd.DataFrame(index=dates)
for i, row in sample_space_df.iterrows():
    mean_val = row['p mean']
    std_val = row['p standard deviation']
    synthetic = np.random.normal(loc=mean_val, scale=std_val, size=n)
    while np.any(synthetic <= 0):
        synthetic[synthetic <= 0] = np.random.normal(loc=mean_val, scale=std_val, size=np.sum(synthetic <= 0))

    # make 13% to get recharge, recharge = 13% of precipitation
    p_time_series[f'precip{i}'] = synthetic
    rech_timeseries[f'rech{i}'] = synthetic * 0.13

p_time_series.to_excel(f'p_sequence.xlsx', index=False)
rech_timeseries.to_excel(f'r_sequence.xlsx', index=False)

#temperature
t_time_series = pd.DataFrame(index=dates)
for i, row in sample_space_df.iterrows():
    mean_val = row['t mean']
    std_val = row['t standard deviation']
    synthetic = np.random.normal(loc=mean_val, scale=std_val, size=n)
    while np.any(synthetic <= 0):
        synthetic[synthetic <= 0] = np.random.normal(loc=mean_val, scale=std_val, size=np.sum(synthetic <= 0))

    t_time_series[f'temp{i}'] = synthetic

t_time_series.to_excel(f't_sequence.xlsx', index=False)

#show changes in sea level rise
low = 0.4627106
midlow = 0.7957106
mid = 1.237711
midhigh = 1.620711
high = 2.441711
original = 1

#create time series for sea level rises
high1 = []
low1 = []
mid1 = []
midlow1 = []
midhigh1 = []
og = []

o = original/75
oi = o
for i in range(75):
    og.append(o)
    o=o+oi

h = high/75
hi = h
for i in range(75):
    high1.append(h)
    h=h+hi

l = low/75
li = l
for i in range(75):
    low1.append(l)
    l=l+li

m1 = mid/75
m1i = m1
for i in range(75):
    mid1.append(m1)
    m1=m1+m1i

m2 = midlow/75
m2i = m2
for i in range(75):
    midlow1.append(m2)
    m2=m2+m2i

m3 = midhigh/75
m3i = m3
for i in range(75):
    midhigh1.append(m3)
    m3=m3+m3i



#precip, temp, slr sequences/time series plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12,6))

ax1.plot(dates, p_time_series, color='navy', alpha=0.3)
ax1.axhline(y=1111, color='maroon', alpha=0.7, linewidth=2)
ax1.set_ylabel('precipitation (mm)')
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')

ax2.plot(dates, t_time_series, color='navy', alpha=0.3)
ax2.axhline(y=13.55, color='maroon', alpha=0.7, linewidth=2)
ax2.set_ylabel('temperature (°C)')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')

#plot the five sea level rise scenarios
ax3.plot(dates, high1, label='high', color='navy', linewidth=2, alpha=0.8)
ax3.plot(dates, midhigh1, label='medium-high', color='navy', linewidth=2, alpha=0.8)
ax3.plot(dates, mid1, label='medium', color='navy', linewidth=2, alpha=0.8)
ax3.plot(dates, midlow1, label='medium-low', color='navy', linewidth=2, alpha=0.8)
ax3.plot(dates, low1, label='low', color='navy', linewidth=2, alpha=0.8)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position('right')
ax3.set_ylabel('sea level rise (m)')
ax3.set_xlabel('projected years')
plt.legend().set_visible(False)
plt.show()
