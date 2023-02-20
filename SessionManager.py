# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:47:05 2022

@author: NRCKI-261

Session manager
"""

import shutil
import os
from datetime import datetime

session_name = 'T10_22_230_1'
mode = 'OH'

#%% Pathes
regimes = 'Regimes' #main folder name

lists = 'lists' #folder name with lists
objects = 'objects' #folder name with cached objects

sigview = 'sigview_lists' #files from CreateList.py (will be only inside lists)
maps = '2d_maps' #files from 2D_t10.py
ne_sort = 'ne_sort' #files from ne_sort.py
phi_profiles = 'Phi_profiles' #files from Phi_profiles.py
plot_carpets = 'Plot_carpets' #files from Plot_carpets.py

#%% Functions
def create_session(sname, mode):
    pathes = []
    #create folder session name in Regimes
    #create lists and objects inside
    #inside of lists create folders
    for modules in sigview, maps, ne_sort, phi_profiles, plot_carpets:
        os.makedirs(os.path.join(regimes, sname, mode, lists, modules), exist_ok=True)
        pathes.append(os.path.join(regimes, sname, mode, lists, modules))
        
    #inside of objects create folders
    for modules in maps, ne_sort, phi_profiles, plot_carpets:
        os.makedirs(os.path.join(regimes, sname, mode, objects, modules), exist_ok=True)
        pathes.append(os.path.join(regimes, sname, mode, lists, modules))
    
    save_session(sname, mode, pathes)
    
    return pathes

def save_session(sname, mode, pathes):
    time_format = '%d-%m-%Y %H:%M:%S'
    
    with open(os.path.join(regimes, 'Sessions.txt')) as file:
        contents = file.readlines()
        if sname + ', ' + mode+'\n' in contents:
            raise NameError('Value already exist')
                
    with open(os.path.join(regimes, 'Sessions.txt'), 'a') as file:
        file.write(datetime.now().strftime(time_format) + '\n' +
                   sname + ', ' + mode + '\n\n')
    
    #save actual pathes
    with open(os.path.join(regimes, sname, mode, 'Pathes.txt'), 'w') as file:
        file.write(datetime.now().strftime(time_format) + '\n')
        for path in pathes:
            file.write(path+'\n')

print(create_session(session_name, mode))
print(len(create_session(session_name, mode)))