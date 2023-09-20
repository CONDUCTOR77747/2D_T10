# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:32:13 2023

@author: ammosov_yam
"""

import matplotlib.pyplot as plt
import grid_functions as grid
import os

parameters = {
    #number of trajectories
    'N_start': 980,
    'N_stop': 999,
    #Beam energy
    'E_start': 180,
    'E_stop': 330,
    'E_step': 10,
    #Slit numbers
    'slit_start': 3,
    'slit_stop': 3,
    #Number of thin filaments
    'n_filaments': 3,
    'n_dim': 5000,
              }

language='ru'

mode = 'home'
if mode=='home':
    dirname1 = "G:\\Другие компьютеры\\PC-NRCKI\\radref_calc_T-10\\output\\B22_I230_offax1"
    dirname2 = 'G:\\Другие компьютеры\\PC-NRCKI\\radref_calc_T-10\\output'
    # dirname2 = 'H:\\Другие компьютеры\\PC-NRCKI\\radref_calc_T-10\\output\\SGTM1'
elif mode=='nrcki':
    dirname1 = "C:\\Progs\\Py\\Coils\\T-10_1Dbeam_NewAnalyzer_5slits\\output\\OLD\\B22_I230_offax1"
    dirname2 = "C:\\Progs\\Py\\Radreferences_T-10\\radref_calc_T-10\\output"
    # dirname2 = "C:\\Progs\\Py\\Radreferences_T-10\\radref_calc_T-10\\output\\SGTM1"
    
# plot experiment
traj0 = grid.import_traj(parameters, dirname1)
fig, ax = plt.subplots()
grid.plot_t10_contour(ax)

# grid.plot_grid(ax, traj0, legend=False, marker_A2=None, marker_E=None,
#                 A2=False, color_E='red', linewidth=5, language=language)

#plot calc
traj0 = grid.import_traj(parameters, dirname2)
grid.plot_grid(ax, traj0, legend=False, marker_A2=None, marker_E=None,
               A2=True, alpha_E=1, grid=False, linewidth_E=5, language=language,
               color_A2='blue', alpha_A2=0.7, linewidth_A2=3, color_E='green', cap_style='round')

# № №№ ### grid.delete_files_with_number(dirname2, 220) ## ;;№№№