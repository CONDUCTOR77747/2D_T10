# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:29:55 2019

@author: user
"""
import numpy as np
import pandas as pd
import import_data as imd
import re
import matplotlib.pyplot as plt
# %% Operation
slit = imd.slit
ne_deviation = 25 #percentage % of deviation from mean ne
filename = imd.filename_ne_sort_load
path_to_save = imd.filename_ne_sort_save_obj
# %% Import Data
print('\nimporting data\n')
print('slit: ', slit)
print('shot list: ', filename)

# Load list of shots and energies
amount_of_shots, shots, energies, time_intervals = imd.load_shots(filename)
ne, list_x, list_y, list_rho, list_Ua2 = [], [], [], [], []
mode = 'pickle'
res = np.zeros([0, 8])

# regex get time intervals
# re.findall(r"(\d+[.]+\d*)", time_intervals[i])

# load signals from database
for i in range(amount_of_shots):   
    print(f'\n{i+1}/{amount_of_shots}')
    alpha2 = imd.load_signals(mode, 'A2', shots[i], slit, time_intervals[i], 
                              imd.filename_ne_sort_save_obj)
    dens = imd.load_signals(mode, 'ne', shots[i], slit, time_intervals[i],
                            imd.filename_ne_sort_save_obj)
    ne_mean = np.mean(dens.y)
    print('shot #{}, E={}'.format(shots[i], energies[i]))
    print('<ne> = {:.3f}'.format(ne_mean))
    print(f'time {time_intervals[i]}')
    ne.append(ne_mean)
    
    fname = imd.load_radrefs(shots[i], slit, energies[i], 'file')
    print('rho: ', fname)
    print('pickle_path: ', imd.filename_ne_sort_save_obj)

    # [0]A2 [1]B3 [2]A3 [3]B2 [4]rho [5]x [6]y [7]z
    radref = np.genfromtxt(fname, skip_header=1)

    for j in np.arange(0, radref.shape[0]):
        rho = radref[j, 4]
        Ua2 = radref[j, 0]
        
        if (Ua2 >= min(alpha2.y)) & (Ua2 <= max(alpha2.y)):
            x = radref[j, 5]
            y = radref[j, 6]
            
            res = np.append(res, [[int(shots[i]), int(energies[i]), 
                        time_intervals[i], float(ne_mean),
                        float(x), float(y),float(rho), float(Ua2)]], axis=0)
            
# creating DataFrame
df = pd.DataFrame()
df['Shot'] = res[:,0]
df['Ebeam'] = res[:,1]
df['time_interval'] = res[:,2]
df['ne'] = res[:,3]
df['x'] = res[:,4]
df['y'] = res[:,5]
df['rho'] = res[:,6]
df['Ua2'] = res[:,7]

df['ne'] = pd.to_numeric(df['ne'])
df['x'] = pd.to_numeric(df['x'])
df['y'] = pd.to_numeric(df['y'])
df['rho'] = pd.to_numeric(df['rho'])
df['Ua2'] = pd.to_numeric(df['Ua2'])

ebeam_tot = len(set(df['Ebeam']))
print(f'TOTAL AMOUNT OF ENERGIES: {ebeam_tot}')

# sort by different ne (baskets)
bins_ne = []
baskets_ne = np.zeros([0, 4])
dataframe_collection = {}
ne_min = 0.6
ne_max = 3.0
ne_step = 0.05
eps = 0.2
counter = 0
n_range = np.arange(ne_min, ne_max, ne_step)
for n in n_range:
    df_basket = df[abs(df['ne'] - n) <= eps]
    dataframe_collection[n] = df_basket
    ne_mean = df_basket['ne'].mean()
    
    amount_of_Ebeams = len(set(df_basket['Ebeam']))
    df_len = len(df_basket)
    
    print(f'{[counter]}', 'ne = {:.2f}, E:'.format(n), amount_of_Ebeams,
            f', len = {df_len}', ', ne_mean = {:.3f}'.format(ne_mean))
    
    baskets_ne = np.append(baskets_ne, [[n, amount_of_Ebeams, df_len, 
                                         ne_mean]], axis=0)
    bins_ne.append(n)
    
    counter = counter + 1

# get list of xticks
list_xticks = [round(x - ne_step/2., 2) for x in bins_ne]

# Histogram plot
fig, ax = plt.subplots()
textsize = 15
plt.hist(df['ne'], bins=bins_ne, edgecolor='black', linewidth=1.2)
plt.xticks(list_xticks, size=textsize)
ax.set_xticklabels(list_xticks, rotation=40)
plt.yticks(size=textsize)
plt.xlabel('ne', size=textsize)
plt.ylabel('amount of scans', size=textsize)
plt.grid()
plt.show

# info about ne baskets
df_tb = pd.DataFrame()
df_tb['ne'] = baskets_ne[:,0]
df_tb['№_Ebeams'] = baskets_ne[:,1]
df_tb['№_points'] = baskets_ne[:,2]
df_tb['ne_mean'] = baskets_ne[:,3]

# sort values to get max amount of Ebeams and points and get index with needed ne
df_tb.sort_values(['№_Ebeams', '№_points'], ascending=[False, False], inplace=True)
df_index = df_tb.index[0]

# sort by time interval
df_res = dataframe_collection[n_range[df_index]]
df_res.drop_duplicates(['x','y'], inplace=True)