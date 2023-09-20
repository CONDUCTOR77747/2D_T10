# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:47:05 2022

@author: NRCKI-261

Session manager
"""

import os
from datetime import datetime

device = ['t10', 't15', 'tjII']
session_name = 'T10_22_230_1'
mode = 'OH'

#%% paths
regimes = 'Regimes' #main folder name

lists = 'lists' #folder name with lists
objects = 'objects' #folder name with cached objects

sigview = 'sigview_lists' #files from CreateList.py (will be only inside lists)
maps = '2d_maps' #files from 2D_t10.py
ne_sort = 'ne_sort' #files from ne_sort.py
phi_profiles = 'Phi_profiles' #files from Phi_profiles.py
plot_carpets = 'Plot_carpets' #files from Plot_carpets.py

#%% Functions

class Regime():
    
    def __init__(self, name, B, I):
        self.name = name
        self.B = B
        self.I = I
        
        paths = []
        #create folder session name in Regimes
        #create lists and objects inside
        #inside of lists create folders
        for modules in sigview, maps, ne_sort, phi_profiles, plot_carpets:
            os.makedirs(os.path.join(regimes, name, mode, lists, modules), exist_ok=True)
            paths.append(os.path.join(regimes, name, mode, lists, modules))
            
        #inside of objects create folders
        for modules in maps, ne_sort, phi_profiles, plot_carpets:
            os.makedirs(os.path.join(regimes, name, mode, objects, modules), exist_ok=True)
            paths.append(os.path.join(regimes, name, mode, lists, modules))
        
        self.save(name, mode, paths)

    def save(self, name, mode, paths):
        time_format = '%d-%m-%Y %H:%M:%S'
        
        with open(os.path.join(regimes, 'Sessions.txt')) as file:
            contents = file.readlines()
            if name + ', ' + mode+'\n' in contents:
                raise NameError('Value already exist')
                    
        with open(os.path.join(regimes, 'Sessions.txt'), 'a') as file:
            file.write(datetime.now().strftime(time_format) + '\n' +
                       name + ', ' + mode + '\n\n')
        
        #save actual paths
        with open(os.path.join(regimes, name, mode, 'paths.txt'), 'w') as file:
            file.write(datetime.now().strftime(time_format) + '\n')
            for path in paths:
                file.write(path+'\n')
    