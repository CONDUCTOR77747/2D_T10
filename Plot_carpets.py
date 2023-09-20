# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:25:26 2022

@author: NRCKI-261
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import import_data as imd
import copy
from datetime import datetime
from shapely.geometry import LineString, Point
import pickle
import matplotlib.ticker as ticker
from matplotlib import cm
import re

#%%
def plot_carpet(shot, signal, time_interval):
    
    vmin = -100
    vmax = -60
    nfft = 2048
    
    fig1 = plt.figure(figsize=(10,5))
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212, sharex = ax1)
    
    ax1.set_title(shot)
    ax1.plot(signal.x, signal.y, linewidth=0.1)
    
    spec = ax2.specgram(signal.y, Fs=1e6, NFFT = nfft, noverlap=int(nfft/2),  
                        cmap = 'jet', scale = 'dB', mode='psd', xextent=(0,998.9), pad_to=nfft*2, vmin=vmin, vmax=vmax)
    
    ax1.axvspan(float(time_interval[0]), float(time_interval[1]), 
                facecolor="magenta", edgecolor='black', linewidth=1, alpha=0.3, picker=5)
    ax2.axvspan(float(time_interval[0]), float(time_interval[1]), 
                facecolor="magenta", edgecolor='black', linewidth=1, alpha=0.3, picker=5)
    
    ti_center = float(time_interval[0]) + abs(float(time_interval[0]) - float(time_interval[1]))/2.
    
    def on_pick(event):
        x_shift = 150
        ax1.set_xlim(ti_center-x_shift, ti_center+x_shift)
        ax2.set_ylim(0, 100000)
        print('zoomed in')
        fig1.canvas.draw()
    
    cid = fig1.canvas.mpl_connect('pick_event', on_pick)
    
    ax1.grid()
    ax2.grid()
    plt.show()

# %% Import Data
# PATH
slit = imd.slit
path_list = imd.path_plot_carpets_list
path_obj = imd.path_plot_carpets_save_obj
path_img = imd.path_plot_carpets_save_img

# Flags
# polynom_plot_flag = 0
# flag_save_list = 0

flag_plot_carpets = 1 #plot spectrograms
flag_plot_only_first_carpet = 0
flag_save_carpets = 0
log_colorbar_flag= 0

# flag_save_origin = 0
# flag_save_multicarpet = 0

print('\nimporting data\n')
print('slit: ', slit)
print('shot list: ', path_list)

# Load list of shots and energies
amount_of_shots, shots, energies, time_intervals = imd.load_shots(path_list)
list_Phi_x, list_Phi_y, list_Rho_x, list_Rho_y, ne = [], [], [], [], []
time_interval = ''

mode = 'pickle'

# load signals from database
for i in range(amount_of_shots):   
    print(f'\n{i+1}/{amount_of_shots}')
    
    radref = imd.load_radrefs(shots[i], slit, energies[i])
    
    print(shots[i], f'E = {energies[i]}', slit, f'\nrho: {radref}')
    
    #Phi = imd.load_signals(mode, 'Phi', shots[i], slit, time_intervals[i], path_obj)
    PhiRaw = imd.load_signals(mode, 'PhiRaw', shots[i], slit, time_interval, path_obj)
    # Radius = imd.load_signals(mode, 'Radius', shots[i], slit, 
    #                           time_intervals[i], path_obj, radref)
    # dens = imd.load_signals(mode, 'ne', shots[i], slit, time_intervals[i],
    #                         path_obj)
    
    # ne_mean = str(round(np.mean(dens.y), 3))
    # ne.append(ne_mean)
    
    print('shot: #{}, E={}'.format(shots[i], energies[i]))
    print(f'time: {time_intervals[i]}')
    print('pickle: ', path_obj)

print('loaded: ', path_list)

pattern = r"\d\d\d\.\d\d"
time_interval = re.findall(pattern, time_intervals[i])
#%% plot spectrogramm

shot = 73191
plot_carpet(shot, PhiRaw, time_interval)