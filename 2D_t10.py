# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:29:55 2019

@author: user
"""
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import interpolate
import import_data as imd
import pandas as pd
import mplcursors
import sys
import copy
import matplotlib.patheffects as path_effects
from shapely.geometry import LineString, Point

#%% Functions

def bin_mean(arr, bin_size=1):
    if bin_size == 1:
        extended_arr = arr
    else:
        extended_arr = np.pad(arr.astype(float),
                              (0, bin_size - len(arr) % bin_size),
                              mode='constant', constant_values=np.nan)
    return np.nanmean(extended_arr.reshape(-1, bin_size), axis=1)

def get_signal(signal_name, df):
    """
    Parameters
    ----------
    signal_name : string
        Phi, RMSPhi, Itot, RMSItot, RelRMSItot
    res : 2d array of float64
        [0]x [1]y [2]rho [3]Ua2 [4]Phi [5]RMSPhi [6]Itot
        [7]RMSItot [8]RelRMSItot [9]Ebeam [10]Zd

    Returns
    -------
    1d array of float64, string, float, float
        Array with signal, signal title, colorbar min value, colorbar max value

    """
    if signal_name == "Phi":
        return df['Phi'].to_numpy(), r'$\varphi$, кВ', -1.5, 0.4 #-0.5, 0.5
    elif signal_name == "RMSPhi":
        return df['RMSPhi'].to_numpy(), 'RMS '+r'$\varphi$, кВ',  0, 0.15
    elif signal_name == "Itot":
        return df['Itot'].to_numpy(), 'Itot',  -1.0, 1.0
    elif signal_name == "RMSItot":
        return df['RMSItot'].to_numpy(), 'RMS Itot',  -0.05, 0.35
    elif signal_name == "RelRMSItot":
        return df['RelRMSItot'].to_numpy(), r'RelRMS($\delta I_{tot}/I_{tot}$)',  0.01, 0.05
    
# check if two points intersects
def xy_intersec(x1,y1,x2,y2,eps):
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    if dist <= eps:
        # print('Intersect')
        return True
    else:
        # print('NOT Intersect')
        return False
# %% Operation
histogramm_flag = 0
# PATH
# mode = 0 load list without sort
# mode = 1 load sorted list via ne and phi profiles
# mode = 2 plot histogramm
mode = 1

if mode == 0:
    path_load_list = imd.path_2d_t10_load # load list without sort
    save_flag = 1 # save data for phi profiles
elif mode == 1:
    path_load_list = imd.path_2d_t10_save # load sorted list via ne and phi profiles
    save_flag = 0 # save data for phi profiles
elif mode == 2:
    path_load_list = imd.path_2d_t10_load # load list without sort
    path_load_list_sorted = imd.path_2d_t10_save # load sorted list via ne and phi profiles
    save_flag = 0 # save data for phi profiles
    histogramm_flag = 1

path_save_list = imd.path_2d_t10_save
path_obj = imd.path_2d_t10_save_obj

xy_eps = 0.3 # dots interceprion
ne_eps = 0.2 #+-0.1
interpolation_flag = 1
show_dots_flag = 1
show_title_flag = 0
colorbar_flag = 1
log_colorbar_flag = 0
grid_flag = 0
sort_ne_intdots_zd_flag = 1
signal_name = 'PhiRaw' # Phi PhiRaw RMSPhi Itot RMSItot RelRMSItot

sort_zd_flag = 1
z_min_val, z_max_val = -0.75, 0.75 # Zd filter (mask)

phi_profiles_flag = 1
phi_profiles_mode = 'ne_colorbar' #time_intervals, shots, ebeams

slit = imd.slit
# %% Import Data
print('\nimporting data\n')
print('slit: ', slit)
print('shot list: ', path_load_list)

# Load list of shots and energies
amount_of_shots, shots, energies, time_intervals = imd.load_shots(path_load_list)

# define result array
res = np.zeros([0, 14])

# mode = 'loader' - load signals via SigView Loader AND pickle them to a file
# mode = 'pickle' - load signals from pickled files path: objects/%shot%_file.obj or if there is not - load files
mode = 'pickle'

ne_list = []

# load signals from database
for i in range(len(shots)):
    
    print(f'\n{i+1}/{amount_of_shots}')
    
    dens = imd.load_signals(mode, 'ne', shots[i], slit, time_intervals[i], path_obj)
    alpha2 = imd.load_signals(mode, 'A2', shots[i], slit, time_intervals[i], path_obj)
    alpha2Raw = imd.load_signals(mode, 'A2Raw', shots[i], slit, time_intervals[i], path_obj)
    Itot = imd.load_signals(mode, 'Itot', shots[i], slit, time_intervals[i], path_obj)
    RMSItot = imd.load_signals(mode, 'RMSItot', shots[i], slit, time_intervals[i], path_obj)
    RelRMSItot = imd.load_signals(mode, 'RelRMSItot', shots[i], slit, time_intervals[i], path_obj)
    Phi = imd.load_signals(mode, 'Phi', shots[i], slit, time_intervals[i], path_obj)
    PhiRaw = imd.load_signals(mode, 'PhiRaw', shots[i], slit, time_intervals[i], path_obj)
    RMSPhi = imd.load_signals(mode, 'RMSPhi', shots[i], slit, time_intervals[i], path_obj)
    Zd = imd.load_signals(mode, 'Zd', shots[i], slit, time_intervals[i], path_obj)
    ZdRaw = imd.load_signals(mode, 'ZdRaw', shots[i], slit, time_intervals[i], path_obj)
    
    ne_mean = np.mean(dens.y)
    ne_list.append(ne_mean)
    print('shot #{}, E={}'.format(shots[i], energies[i]))
    print('<ne> = {:.3f}'.format(ne_mean))
    print(f'time {time_intervals[i]}')

    # correct Phi according to line averaged density
    # dens_interp = interpolate.interp1d(dens.x, dens.y)
    # Phi.y = Phi.y * dens_interp(Phi.x) / dens_base
    
    if signal_name == 'PhiRaw':
        # make Ua2 interpolants
        inds = alpha2Raw.y.argsort()
        
        # Interpolate data sets according to A2 points number
        Phi_interp = interpolate.interp1d(bin_mean(alpha2Raw.y[inds]),
                                          bin_mean(PhiRaw.y[inds]),
                                          bounds_error=False)

        Zd_interp = interpolate.interp1d(bin_mean(alpha2Raw.y[inds]),
                                         bin_mean(ZdRaw.y[inds]),
                                         bounds_error=False)
    else:
        # make Ua2 interpolants
        inds = alpha2.y.argsort()
        
        # Interpolate data sets according to A2 points number
        Phi_interp = interpolate.interp1d(bin_mean(alpha2.y[inds]),
                                          bin_mean(Phi.y[inds]),
                                          bounds_error=False)
        RMSPhi_interp = interpolate.interp1d(bin_mean(alpha2.y[inds]),
                                              bin_mean(RMSPhi.y[inds]),
                                              bounds_error=False)   
        Itot_interp = interpolate.interp1d(bin_mean(alpha2.y[inds]),
                                           bin_mean(Itot.y[inds]),
                                           bounds_error=False)
        RMSItot_interp = interpolate.interp1d(bin_mean(alpha2.y[inds]),
                                              bin_mean(RMSItot.y[inds]),
                                              bounds_error=False)
        RelRMSItot_interp = interpolate.interp1d(bin_mean(alpha2.y[inds]),
                                                 bin_mean(RelRMSItot.y[inds]),
                                                 bounds_error=False)
        Zd_interp = interpolate.interp1d(bin_mean(alpha2.y[inds]),
                                         bin_mean(Zd.y[inds]),
                                         bounds_error=False)

    # get discrete Ua2, rho and x,y values from radref files
    fname = imd.load_radrefs(shots[i], slit, energies[i], 'file')
    print('rho: ', fname)
    print('pickle_path: ', path_obj)

    # [0]A2 [1]B3 [2]A3 [3]B2 [4]rho [5]x [6]y [7]z
    radref = np.genfromtxt(fname, skip_header=1)

    for j in np.arange(0, radref.shape[0]):
        rho = radref[j, 4]
        Ua2 = radref[j, 0]
        
        if (Ua2 >= min(alpha2.y)) & (Ua2 <= max(alpha2.y)):
            x = radref[j, 5]
            y = radref[j, 6]

            res = np.append(res, [[shots[i], energies[i], time_intervals[i], 
                                   ne_mean, x, y, rho, Ua2, Phi_interp(Ua2),
                                  RMSPhi_interp(Ua2), Itot_interp(Ua2),
                                  RMSItot_interp(Ua2), RelRMSItot_interp(Ua2),
                                  Zd_interp(Ua2)]], axis=0)

print('\nloaded:', path_load_list)
#%% Pass the data to a pandas DataFrame
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
df['Phi'] = res[:,8]
df['RMSPhi'] = res[:,9]
df['Itot'] = res[:,10]
df['RMSItot'] = res[:,11]
df['RelRMSItot'] = res[:,12]
df['Zd'] = res[:,13]
# convert string to numeric data
df['Ebeam'] = pd.to_numeric(df['Ebeam'])
df['ne'] = pd.to_numeric(df['ne'])
df['x'] = pd.to_numeric(df['x'])
df['y'] = pd.to_numeric(df['y'])
df['rho'] = pd.to_numeric(df['rho'])
df['Ua2'] = pd.to_numeric(df['Ua2'])
df['Phi'] = pd.to_numeric(df['Phi'])
df['RMSPhi'] = pd.to_numeric(df['RMSPhi'])
df['Itot'] = pd.to_numeric(df['Itot'])
df['RMSItot'] = pd.to_numeric(df['RMSItot'])
df['RelRMSItot'] = pd.to_numeric(df['RelRMSItot'])
df['Zd'] = pd.to_numeric(df['Zd'])

# save copy of raw data
df_imported = copy.deepcopy(df)

#%% Plot Background (T-10 Grid)
# Import T-10 camera data
camera = imd.camera

#plot camera and plasma contour
fig, ax = plt.subplots()

# camera = camera/100 # [m]
ax.plot(camera[:,0],camera[:,1])
a = 0.3*100 #plasma radius [cm]
Del = -0.01 #plasma shift [m]
plasma_contour = plt.Circle((Del, 0), a, color='r', linewidth = 1.5, fill=False)
ax.add_artist(plasma_contour)

#plot circular magnetic surfaces
dr = 0.01 # radius step [m]
dx = dr/5 # x step
R = np.arange(0,0.3+dr,dr) # radius range in [m] 
dfi = math.pi/50
for i in R:
    surf = np.zeros([0,2])
    for fi in np.arange(0,2*math.pi+dfi,dfi):
        surf = np.append(surf,[[i*np.cos(fi),i*np.sin(fi)]],axis = 0)
#    surf = np.append(surf,[[i+Del,0]],axis = 0)
    surf = np.append(surf,[[Del,0]],axis = 0)
    if 100*i % 5 == 0:
        ax.plot(surf[:,0]*100 ,surf[:,1]*100, linewidth=2, color='k')
    else:
        ax.plot(surf[:,0]*100 ,surf[:,1]*100, linewidth=1, color='k')
#%% Sort by ne
if sort_ne_intdots_zd_flag:
    ebeam_tot = len(set(df['Ebeam']))
    print(f'\nTOTAL AMOUNT OF ENERGIES: {ebeam_tot}\n')
    
    bins_ne = []
    baskets_ne = np.zeros([0, 4])
    # sort by different ne (baskets)
    dataframe_collection_ne = {}
    ne_min = 0.6
    ne_max = 2.8
    ne_step = 0.1
    counter = 0
    n_range = np.arange(ne_min, ne_max, ne_step)
    for n in n_range:
        df_basket = df[abs(df['ne'] - n) <= ne_eps]
        dataframe_collection_ne[n] = df_basket
        ne_mean = df_basket['ne'].mean()
        
        amount_of_Ebeams = len(set(df_basket['Ebeam']))
        df_len = len(df_basket)
        
        print(f'{[counter]}', 'ne = {:.2f}, E:'.format(n), amount_of_Ebeams,
                f', len = {df_len}', ', ne_mean = {:.3f}'.format(ne_mean))
        
        baskets_ne = np.append(baskets_ne, [[n, amount_of_Ebeams, df_len, 
                                              ne_mean]], axis=0)
        bins_ne.append(n)
        
        counter = counter + 1
     
    #%% Histogram plot
    if histogramm_flag:
        fig2, ax2 = plt.subplots()
        # get list of xticks
        list_xticks = [round(x-ne_step/2., 2) for x in bins_ne]
        textsize = 35
        N, bins, patches = ax2.hist(df['ne'].drop_duplicates(), bins=bins_ne, edgecolor='black', linewidth=1.2)
        ax2.set_xticks(list_xticks)
        ax2.set_xticklabels(list_xticks, rotation=90)
        ax2.tick_params(axis='both', labelsize = textsize)
        ax2.set_xlabel('ne', size=textsize)
        ax2.set_ylabel('amount of scans', size=textsize)
        ax2.grid()
    #%% info about ne baskets
    
    df_tb = pd.DataFrame()
    df_tb['ne'] = baskets_ne[:,0]
    df_tb['№_Ebeams'] = baskets_ne[:,1]
    df_tb['№_points'] = baskets_ne[:,2]
    df_tb['ne_mean'] = baskets_ne[:,3]
    
    # sort values to get max amount of Ebeams and points and get index with needed ne
    df_tb.sort_values(['№_Ebeams', '№_points'], ascending=[False, False], inplace=True)
    df_index = df_tb.index[0]
    print(f'Index: {df_index} was taken.')
    
    # create dataframe
    df = dataframe_collection_ne[n_range[df_index]]
    ne_mean = df['ne'].mean()
    #%% delete interceprions between dots
    list_ebeam = sorted(list(set(df['Ebeam'])))
    dataframe_collection_ebeam = {}
    dataframe_collection_ebeam_dropped = {}
    counter = 0
    for ebeam in list_ebeam:
        df_basket_E = df[df['Ebeam'] == ebeam]
        dataframe_collection_ebeam[ebeam] = df_basket_E
        df_len = len(df_basket_E)
        print(f'{[counter]} E: {ebeam}, len = {df_len}',)
        counter = counter + 1
    
    print(f'Total {len(df)} dots.')
    
    for ebeam in list_ebeam:
        list_dropped = []
    
        df = dataframe_collection_ebeam[ebeam]
        for i in df.index:
            for j in df.index:
                if i != j and (i not in list_dropped) and (j not in list_dropped):
                    if xy_intersec(df['x'].loc[i], df['y'].loc[i], 
                                  df['x'].loc[j], df['y'].loc[j], xy_eps):
                        if abs(ne_mean-df['ne'].loc[i]) <= abs(ne_mean-df['ne'].loc[j]):
                            list_dropped.append(j)
                        else:
                            list_dropped.append(i)
    
        df_dropped = df.drop(list_dropped)
        dataframe_collection_ebeam_dropped[ebeam] = df_dropped
    
    df = pd.concat(dataframe_collection_ebeam_dropped)
    print(f'Sorted by n and E. Total {len(df)} dots.')

    #%% Z mask
    if sort_zd_flag:
        df = df.loc[(df.Zd >= z_min_val) & (df.Zd <= z_max_val)]
        print(f'Sorted by Zd. Total {len(df)} dots.')
#%% save list with defined ne without intercepted dots
df_csv = df.drop_duplicates(subset=['time_interval'])
if save_flag:
    df_csv.to_csv(path_save_list, sep='!', 
header=False, index=False, columns=['Shot', 'Ebeam', 'time_interval'])
    print('saved:', path_save_list)

#%% Plotting

text_size = 35

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.titlesize'] = text_size-27
plt.rcParams['axes.labelsize'] = text_size
plt.rcParams['xtick.labelsize'] = text_size
plt.rcParams['ytick.labelsize'] = text_size
plt.rcParams['legend.fontsize'] = text_size-5
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['axes.grid.axis'] = "both"
# plt.rcParams['figure.figsize'] = (5, 10)

#%% Plot Phi profiles
if phi_profiles_flag:
    fig, ax_phi_prof = plt.subplots()
    lines = []
    
    if phi_profiles_mode == "mono":
    
        x = df['rho'].to_list()
        
        y = df['Phi'].to_list()
        
        for i in range(len(x)):
            line = ax_phi_prof.errorbar(x, 
                                        y, 
                                        xerr=0.5, yerr=0.05, fmt='o',
                                        ecolor='gray', elinewidth=0.5, capsize=4, 
                                        capthick=0.5,
                    label=f'Shot: {0}\nE: {energies[i]}\nTI: {0}\nne: {0}')
    
    
    if phi_profiles_mode == "ne_colorbar":   
        cm = plt.cm.get_cmap('jet')
    
        x = df['rho']
        
        y = df['Phi']
        
        error = np.ones(len(x))*0.05
        
    #     line = ax_phi_prof.errorbar(x, 
    #                                 y, 
    #                                 xerr=0.5, yerr=0.05, fmt='o',
    #                                 ecolor='gray', elinewidth=0.5, capsize=4, 
    #                                 capthick=0.5,
    # label=f'Shot: {shots[i]}\nE: {energies[i]}\nTI: {0}\nne: {0}')
        
        sc = plt.scatter(x,y,s=0,c=df['ne'])
    
    #create colorbar according to the scatter plot
        clb = plt.colorbar(sc)
        
        import matplotlib
        import matplotlib.cm as cm
        norm = matplotlib.colors.Normalize(vmin=min(df['ne']), vmax=max(df['ne']), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        time_color = np.array([(mapper.to_rgba(v)) for v in df['ne']])
        
        #loop over each data point to plot
        for x1, y1, e, color in zip(x, y, error, time_color):
            plt.plot(x1, y1, 'o', color=color)
            plt.errorbar(x1, y1, e, lw=1, capsize=3, color=color)
            
            cursor = mplcursors.cursor(lines, highlight=True, multiple=True)
    
    # Plot parameters
    
    text_size = 35
    ax_phi_prof.tick_params(axis='both', labelsize=text_size)
    ax_phi_prof.grid()
    #ax_phi_prof.set_xlim(7, 35)
    #ax_phi_prof.set_ylim(-1.6, 0.25)
    plt.subplots_adjust(top=0.9,
    bottom=0.11,
    left=0.26,
    right=0.8,
    hspace=0.2,
    wspace=0.2)
    ax_phi_prof.set_xlabel('r, см', size=text_size)
    ax_phi_prof.set_ylabel(r'$\varphi$, кВ', size=text_size)

#%% Grid Plot

# get colors for plotting grid
def get_color(ebeam):
    colors = []
    ebeams = []
    j = 0
    for i in range(150, 351, 10):
        colors.append(f'C{j}')
        ebeams.append(i)
        j += 1
    for n in range(len(ebeams)):
        if ebeam == ebeams[n]:
            return colors[n]

set_of_energies = sorted(list(set(energies)))
if grid_flag: 
    for flag_grid_one_enegry in set_of_energies:
        ebeams = []
        texts = []
        grouped = df_imported.groupby('time_interval')
        for name in df['time_interval'].unique():
            temp = grouped.get_group(name)
            if flag_grid_one_enegry and int(temp['Ebeam'].unique()) == flag_grid_one_enegry: #need to delete
                grid_plot = ax.plot(temp['x'].to_numpy(), temp['y'].to_numpy(), 
                                    linewidth=7, solid_capstyle='round',
                                    label='#{} {} keV'.format(int(temp['Shot'].unique()[0]), 
                                                              int(temp['Ebeam'].unique())), 
                                    c = get_color(temp['Ebeam'].unique()[0])) #choose color
                texts.append(grid_plot)
    
        grouped = df_imported.groupby('Ebeam')
        #pos = [-0.4,-0.5,-0.5,-0.0,-0.7,-0.5,-0.3,0,0,0.2,0.3,-0.1,-0.2,5]
        for ebeam in df['Ebeam'].unique():
            
            temp = grouped.get_group(ebeam)
            
            if flag_grid_one_enegry and int(temp['Ebeam'].unique()) == flag_grid_one_enegry: #need to delete
                x = temp['x'].to_numpy()[np.argmin(temp['y'].to_numpy())] - 0 #+ pos[j]
                y = np.min(temp['y'].to_numpy()) - 3
                
                ax.annotate(text=str(temp['Ebeam'].unique()[0]), xy=(x, y), 
                            fontsize=text_size-5, rotation=-60, weight='bold', 
                            c=get_color(temp['Ebeam'].unique()[0]), 
                            path_effects=[path_effects.Stroke(linewidth=1.75, foreground='black'),
                               path_effects.Normal()],
                            bbox=dict(boxstyle="square", alpha=1, pad=0, color='white'))
        
    # plt.legend()
#%% Interpolate dots and plot 2d map
xmin=5
xmax=30
ymin=-5
ymax=15
Npoints=50

sig, signal_title, min_val, max_val  = get_signal(signal_name, df)
dots_color = sig #df['Ebeam'].to_numpy()
# min_val = 180
# max_val = 330

cmap = plt.cm.get_cmap("gist_rainbow")

if log_colorbar_flag == 0:
    norm = cm.colors.Normalize(vmax=max_val, vmin=min_val)
elif log_colorbar_flag == 1:
    norm = cm.colors.LogNorm(vmin=0.005, vmax=0.5)

x, y = np.linspace(xmin, xmax, Npoints), np.linspace(ymin, ymax, Npoints)
x, y = np.meshgrid(x, y)

grid = interpolate.griddata((df['x'].to_numpy(), df['y'].to_numpy()), sig, (x,y), method='linear')

labels = []
for i in range(len(df)):
    str_shot = str(df['Shot'].to_numpy()[i])
    str_Ebeam = str(df['Ebeam'].to_numpy()[i])
    str_TI = str(df['time_interval'].to_numpy()[i])
    str_ne = str(round(df['ne'].to_numpy()[i], 3))
    str_Phi = str(round(df['Phi'].to_numpy()[i], 3))
    str_x = str(round(df['x'].to_numpy()[i], 2))
    str_y = str(round(df['y'].to_numpy()[i], 2))
    str_RMSPhi = str(round(df['RMSPhi'].to_numpy()[i], 3))
    str_Itot = str(round(df['Itot'].to_numpy()[i], 3))
    str_RMSItot = str(round(df['RMSItot'].to_numpy()[i], 3))
    str_RelRMSItot = str(round(df['RelRMSItot'].to_numpy()[i], 3))
    str_Zd = str(round(df['Zd'].to_numpy()[i], 3))
    str_RhoEval = str(round(np.sqrt((df['x'].to_numpy()[i])**2+(df['y'].to_numpy()[i])**2),3))
    str_A2 = str(round(df['Ua2'].to_numpy()[i], 3))
    if signal_name == 'Phi':
        label = f'Shot: {str_shot}\nE: {str_Ebeam}\nTI: {str_TI}\nne: {str_ne}\n\
Phi: {str_Phi}\nx = {str_x}\ny = {str_y}\nRho: {str_RhoEval}\nZd: {str_Zd}\n A2: {str_A2}'
    elif signal_name == 'RMSPhi':
        label = f'Shot: {str_shot}\nE: {str_Ebeam}\nTI: {str_TI}\nne: {str_ne}\n\
RMSPhi: {str_RMSPhi}\nx = {str_x}\ny = {str_y}'
    elif signal_name == 'Itot':
        label = f'Shot: {str_shot}\nE: {str_Ebeam}\nTI: {str_TI}\nne: {str_ne}\n\
Itot: {str_Itot}\nx = {str_x}\ny = {str_y}'
    elif signal_name == 'RMSItot':
        label = f'Shot: {str_shot}\nE: {str_Ebeam}\nTI: {str_TI}\nne: {str_ne}\n\
RMSItot: {str_RMSItot}\nx = {str_x}\ny = {str_y}'
    elif signal_name == 'RelRMSItot':
        label = f'Shot: {str_shot}\nE: {str_Ebeam}\nTI: {str_TI}\nne: {str_ne}\n\
RelRMSItot: {str_RelRMSItot}\nx = {str_x}\ny = {str_y}'
    labels.append(label)

if log_colorbar_flag:
    sc = ax.scatter(df['x'].to_numpy(), df['y'].to_numpy(), c=dots_color, norm=norm,
                s=100, cmap=cmap, edgecolors='black')
else:
    sc = ax.scatter(df['x'].to_numpy(), df['y'].to_numpy(), c=dots_color, vmin=min_val, vmax=max_val,
                s=100*show_dots_flag, cmap=cmap, edgecolors='black')
        
    
#%%
mplcursors.cursor(ax).connect(
    "add", lambda sel: sel.annotation.set_text(labels[sel.index]))

if interpolation_flag:
    ax.imshow(grid, interpolation='none',
                origin='lower',
                extent=[xmin, xmax, ymin, ymax],
                norm=norm, cmap=cmap)
    # ax.contourf(grid, interpolation='none',
    #             origin='lower',
    #             extent=[xmin, xmax, ymin, ymax],
    #             norm=norm, cmap=cmap)

if colorbar_flag:
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(label=signal_title, size=text_size)
    cbar.ax.tick_params(labelsize=text_size)

str_ne_mean = str(round(ne_mean, 3))
if show_title_flag:
    ax.set_title(str(len(df)) + f' dots; ne: {str_ne_mean} ± {ne_eps}', fontsize=text_size)


ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x, см', size=text_size)
ax.set_ylabel('y, см', size=text_size)
ax.tick_params(axis='both', labelsize=text_size)
ax.set_xlim(-3, 30)
ax.set_ylim(-3, 30)

ax.grid()
plt.show()