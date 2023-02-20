# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 15:54:40 2022

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
# %% Import Data
# PATH
slit = imd.slit
path_list = imd.path_phi_profiles_list
path_list_outside_the_polygon = imd.path_phi_profiles_list_outside_the_polygon
path_obj = imd.path_phi_profiles_save_obj
path_origin_save = imd.path_phi_profiles_save_origin
path_multicarpet_save = imd.path_phi_profiles_save_multicarpet

# Flags
polynom_plot_flag = 0
flag_save_list = 1
flag_save_origin = 0
flag_save_multicarpet = 0

print('\nimporting data\n')
print('slit: ', slit)
print('shot list: ', path_list)

# Load list of shots and energies
amount_of_shots, shots, energies, time_intervals = imd.load_shots(path_list)
list_Phi_x, list_Phi_y, list_Rho_x, list_Rho_y, ne = [], [], [], [], []
mode = 'pickle'

# load signals from database
for i in range(amount_of_shots):   
    print(f'\n{i+1}/{amount_of_shots}')
    
    radref = imd.load_radrefs(shots[i], slit, energies[i])
    
    print(shots[i], f'E = {energies[i]}', slit, f'\nrho: {radref}')
    
    Phi = imd.load_signals(mode, 'Phi', shots[i], slit, time_intervals[i], path_obj)
    Radius = imd.load_signals(mode, 'Radius', shots[i], slit, 
                              time_intervals[i], path_obj, radref)
    dens = imd.load_signals(mode, 'ne', shots[i], slit, time_intervals[i],
                            path_obj)
    
    ne_mean = str(round(np.mean(dens.y), 3))
    ne.append(ne_mean)
    
    print('shot: #{}, E={}'.format(shots[i], energies[i]))
    print(f'time: {time_intervals[i]}')
    print('pickle: ', path_obj)
    
    list_Phi_x.append(Phi.x)
    list_Phi_y.append(Phi.y)
    list_Rho_x.append(Radius.x)
    list_Rho_y.append(Radius.y)

print('loaded: ', path_list)
#%% Save to Origin
df = pd.DataFrame()
df[f'{shots[0]} Rho']=list_Rho_y[i]
df[f'{shots[0]} Phi']=list_Phi_y[i]
# creating DataFrame
for i in range(amount_of_shots-1):
    df_buff = pd.DataFrame()
    df_buff[f'{shots[i+1]} Rho']=list_Rho_y[i]
    df_buff[f'{shots[i+1]} Phi']=list_Phi_y[i]
    df = pd.concat([df, df_buff], axis=1)
if flag_save_origin:
    df.to_csv(path_origin_save, index=False)
    print(f'Saved to Origin: {path_origin_save}')
    
#%% Save for multicarpet
if flag_save_multicarpet:
    time_format = '%d-%m-%Y %H:%M:%S'
    file_format = '.list'
    save_dir = path_multicarpet_save
    
    itot_file_path = save_dir + 'Itot' + file_format
    phi_file_path = save_dir + 'Phi' + file_format
    
    # creating content of output file
    f_itot = open(itot_file_path, 'a')
    f_itot.write("!"+datetime.now().strftime(time_format)+'\n')
    f_phi = open(phi_file_path, 'a')
    f_phi.write("!" + datetime.now().strftime(time_format) + '\n')
    
    for i in range(amount_of_shots):
        f_itot.write("T10HIBP::Itot{relosc333, slit3, clean, noz, shot" + 
                     str(shots[i]) + ', ' + time_intervals[i] + "} !" + 
                     "{:.3f}".format(float(ne[i])) + " #" + str(shots[i]) + ' E = ' + 
                     str(energies[i]) + '\n')
    
        f_phi.write("T10HIBP::Phi{slit3, clean, noz, shot" + str(shots[i]) + ', ' +
                    time_intervals[i] + "} !" + "{:.3f}".format(float(ne[i])) + " #" 
                    + str(shots[i]) + ' E = ' + str(energies[i]) + '\n')
    
    f_itot.close()
    f_phi.close()
#%% Plot Phi profiles
fig, ax = plt.subplots()
lines = []
for i in range(amount_of_shots):
    line = ax.scatter(list_Rho_y[i], list_Phi_y[i], s=1, 
label=f'Shot: {shots[i]}\nE: {energies[i]}\nTI: {time_intervals[i]}\nne: {ne[i]}')
    lines.append(line)
    
cursor = mplcursors.cursor(lines, highlight=True, multiple=True)

#%% Polynomial fit data

mode = 'loader'

if mode == 'loader':
    list_Phi_flatten = np.hstack(np.array(list_Phi_y, dtype=object))
    list_Rho_flatten = np.hstack(np.array(list_Rho_y, dtype=object))
    
    Rho_min_val = min(list_Rho_flatten)
    Rho_max_val = max(list_Rho_flatten)
    x_poly = np.linspace(Rho_min_val, Rho_max_val)
    
    poly = np.polyfit(list_Rho_flatten, list_Phi_flatten, deg=4)
    # ax.plot(x_poly, np.polyval(poly, x_poly), label='fit', color='r', 
    #         linewidth=3, alpha=1)
    y_poly = np.polyval(poly, x_poly)
    
    coordinate = []
    for i in range(len(y_poly)):
        coordinate.append((x_poly[i],y_poly[i]))
    coordinate = np.array(coordinate)
    
    poly_lines = LineString(coordinate)
    poly = poly_lines.buffer(0.1) #thickness of polygon in kV +- value; poly is polynomial fit polygon
    poly_pickle = open(path_obj+'poly.obj', 'wb')
    pickle.dump(poly, poly_pickle)
    poly_pickle.close()
elif mode == 'pickle':
    poly_pickle = open(path_obj+'poly.obj', 'rb')
    poly = pickle.load(poly_pickle)
    poly_pickle.close()

if polynom_plot_flag:
    x, y = poly.exterior.coords.xy
    ax.plot(x, y, color='black', alpha=1,
        linewidth=4, solid_capstyle='round', zorder=2)
#%% Delete line by pressing Delete button and clear outside of polygon
def delete_data_from_file(filename, data):
    amount_of_shots, shots, energies, time_intervals = imd.load_shots(filename)
    list_dropped = []
    df = pd.DataFrame()
    df['Shot'] = shots
    df['Ebeam'] = energies
    df['time_interval'] = time_intervals
    for val in data:
        print(val)
        for i in df.index:
            if str(df['Shot'].loc[i]) == str(val[0]) \
            and str(df['Ebeam'].loc[i]) == str(val[1]) \
            and str(df['time_interval'].loc[i]) == str(val[2]):
                list_dropped.append(i)
    print(list_dropped)
    df_dropped = df.drop(list_dropped)
    df = df_dropped.reset_index()
    print(df)
    if flag_save_list:
        df.to_csv(path_list, sep='!', 
    header=False, index=False, columns=['Shot', 'Ebeam', 'time_interval'])
        print('saved:', path_list)
    return df

def save_data_to_file(filename, data):
    if flag_save_list:
        f_labels = open(filename, 'w')
        for i in range(len(data)):
            f_labels.write(data[i]+'\n')
        f_labels.close()
        print('saved:', filename)
        
def append_data_to_file(filename, data):
    if flag_save_list:
        f_labels = open(filename, 'a')
        for i in range(len(data)):
            f_labels.write(data[i]+'\n')
        f_labels.close()
        print('saved:', filename)

def on_press(event):
    list_data = []
    row = []
    row_copy = []
    sys.stdout.flush()
    #  Delete line by pressing Delete button
    if event.key == 'delete':
        for sel in cursor.selections:  
            sel.artist.set_visible(False)
            sel.extras[0].set_visible(False)
            sel_label = sel.artist.get_label().split()
            row.append(sel_label[1])
            row.append(sel_label[3])
            row.append(sel_label[5])
            row_copy = copy.deepcopy(row)
            list_data.append(row_copy)
            row.clear()
            cursor.remove_selection(sel)
        delete_data_from_file(path_list, list_data)
        #save deleted scans to file
        data = list_data[0][0] + '!' + list_data[0][1] + '!' + list_data[0][2]
        data1 = []
        data1.append(data)
        append_data_to_file(path_list_outside_the_polygon, data1)
    list_data.clear()
    row_copy.clear()
    
    # Clear outside of polygon
    list_Phi_x_new = []
    list_Phi_y_new = []
    row_x = []
    row_y = []
    list_labels = []
    flag = 0
    if event.key == 'shift+delete':
        # wrtite to file lines inside the polygon
        for i in range(amount_of_shots):
            for j in range(len(list_Phi_y[i])):
                if (poly.contains(Point(list_Rho_y[i][j], list_Phi_y[i][j]))):
                    flag = 1
                    row_x.append(list_Phi_x[i][j])
                    row_y.append(list_Phi_y[i][j])
                elif (row_x and row_y) and flag == 1:                   
                    label = lines[i].get_label().split()
                    label = str(label[1]) + '!' + str(label[3]) + '!from' + \
                    "{:.2f}".format(float(row_x[0])) + 'to' + \
                    "{:.2f}".format(float(row_x[-1]))
                    print(label)
                    list_labels.append(label)
                    list_Phi_x_new.append(row_x)
                    list_Phi_y_new.append(row_y)
                    row_x.clear()
                    row_y.clear()
                    flag = 0
                if (row_x and row_y) and len(list_Phi_y[i])-1 == j and len(list_Phi_y[i])-1 == j and flag==1:
                    label = lines[i].get_label().split()
                    label = str(label[1]) + '!' + str(label[3]) + '!from' + \
                    "{:.2f}".format(float(row_x[0])) + 'to' + \
                    "{:.2f}".format(float(row_x[-1]))
                    
                    print(label)
                    
                    list_labels.append(label)
                    list_Phi_x_new.append(row_x)
                    list_Phi_y_new.append(row_y)
                    row_x.clear()
                    row_y.clear()
        save_data_to_file(path_list, list_labels)
        
        list_Phi_x_new = []
        list_Phi_y_new = []
        row_x = []
        row_y = []
        list_labels = []
        flag = 0
        
        # wrtite to file lines outside the polygon
        for i in range(amount_of_shots):
            for j in range(len(list_Phi_y[i])):
                if not (poly.contains(Point(list_Rho_y[i][j], list_Phi_y[i][j]))):
                    flag = 1
                    row_x.append(list_Phi_x[i][j])
                    row_y.append(list_Phi_y[i][j])
                elif (row_x and row_y) and flag == 1:                   
                    label = lines[i].get_label().split()
                    label = str(label[1]) + '!' + str(label[3]) + '!from' + \
                    "{:.2f}".format(float(row_x[0])) + 'to' + \
                    "{:.2f}".format(float(row_x[-1]))
                    print(label)
                    list_labels.append(label)
                    list_Phi_x_new.append(row_x)
                    list_Phi_y_new.append(row_y)
                    row_x.clear()
                    row_y.clear()
                    flag = 0
                if (row_x and row_y) and len(list_Phi_y[i])-1 == j and len(list_Phi_y[i])-1 == j and flag==1:
                    label = lines[i].get_label().split()
                    label = str(label[1]) + '!' + str(label[3]) + '!from' + \
                    "{:.2f}".format(float(row_x[0])) + 'to' + \
                    "{:.2f}".format(float(row_x[-1]))
                    
                    print(label)
                    
                    list_labels.append(label)
                    list_Phi_x_new.append(row_x)
                    list_Phi_y_new.append(row_y)
                    row_x.clear()
                    row_y.clear()
        save_data_to_file(path_list_outside_the_polygon, list_labels)
        fig.canvas.draw()
        
fig.canvas.mpl_connect('key_press_event', on_press)
#%% Plot parameters
text_size = 35
ax.tick_params(axis='both', labelsize=text_size)
ax.grid()
#ax.set_xlim(7, 35)
#ax.set_ylim(-1.6, 0.25)
plt.subplots_adjust(top=0.9,
bottom=0.11,
left=0.26,
right=0.8,
hspace=0.2,
wspace=0.2)
ax.set_xlabel('Радиус, cm', size=text_size)
ax.set_ylabel('Потенциал, kV', size=text_size)
plt.show()