# -*- coding: utf-8 -*-
"""
@author: NRCKI
"""
import numpy as np
import math
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import interpolate
import import_data as imd
import pandas as pd
import mplcursors
import copy
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
import grid_functions as grid

#%% Language
plot_labels = imd.plot_labels
#%% Functions

def bin_mean(arr, bin_size=1):
    if bin_size == 1:
        extended_arr = arr
    else:
        extended_arr = np.pad(arr.astype(float),
                              (0, bin_size - len(arr) % bin_size),
                              mode='constant', constant_values=np.nan)
    return np.nanmean(extended_arr.reshape(-1, bin_size), axis=1)

def get_signal(signal_name, df, plot_lablels, language):
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
        Array with signal, signal 2de, colorbar min value, colorbar max value

    """
    kv = plot_labels[language]['kV']
    if signal_name == "Phi":
        return df['Phi'].to_numpy(), fr'$\varphi$, {kv}', -0.65, -0.45#-0.5, 0.5 #-1.5, 0.4 #
    elif signal_name == "RMSPhi":
        return df['RMSPhi'].to_numpy(), fr'RMS $\varphi$, {kv}',  0, 0.15
    elif signal_name == "Itot":
        return df['Itot'].to_numpy(), 'Itot',  -1.0, 1.0
    elif signal_name == "RMSItot":
        return df['RMSItot'].to_numpy(), 'RMS Itot',  -0.05, 0.35
    elif signal_name == "RelRMSItot":
        return df['RelRMSItot'].to_numpy(), r'RelRMS($\delta I_{tot}/I_{tot}$)',  0.01, 0.05
    elif signal_name == "A_QCM":
        return df['A_QCM'].to_numpy(), 'A_QCM',  0.00, 0.06
    
# check if two points intersects
def xy_intersec(x1,y1,x2,y2,eps):
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    if dist <= eps:
        # print('Intersect')
        return True
    else:
        # print('NOT Intersect')
        return False
    
# convert cartesian coordinates to polar coordinates
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)
# %% Operation
histogramm_flag = 0
# PATH
# mode = 0 load list without sort
# mode = 1 load sorted list via ne and phi profiles
# mode = 2 plot histogramm
mode = 2

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
    histogramm_flag = 0

path_save_list = imd.path_2d_t10_save
path_obj = imd.path_2d_t10_save_obj
path_origin_save = imd.path_phi_profiles_save_origin
slit = imd.slit
#%% Operational Parameters
language = 'eng'

signal_name = 'A_QCM' # Phi PhiRaw RMSPhi Itot RMSItot RelRMSItot A_QCM
xy_eps = 0.1895 # dots interceprion
ne_eps = 0.2 #+-0.1

""" 2D map """

twoD_plot_flag = 1

show_title_flag = 1
show_dots_flag = 0
show_grid_lines_flag = 0

interpolation_flag = 1
Npoints = 65
#interpolation diapason
#OH
xmin=0
xmax=30
ymin=-3
ymax=19

#offax
# xmin=0
# xmax=24
# ymin=-3
# ymax=19

interpolation_method = "linear" #"cubic", "linear"
imshow_interpolation = 'none'
rescale=False

equipotential_lines_flag = 0
contourf_flag = 0
amount_of_lines = 8

colorbar_flag = 1
cmap_2d_map = 'jet'
log_colorbar_flag = 1

polar_coordinates_flag = 0

""" Grid """

grid_flag = 0
detector_line_mode = "mono" #mono #color #rainbow #one_energy

if detector_line_mode == "mono":
    grid_annotate_flag = 1 # for #mono
else:
    grid_annotate_flag = 0 # for #rainbow #color

# for rainbow connecting dots range is about 0-2 (approximately)
grid_rainbow_threshold = 2 # 0-2 or None for scatter

""" Sorting """

sort_ne_intdots_zd_flag = 1
intdots_flag = 1
sort_zd_flag = 1
z_min_val, z_max_val = -0.85, 0.85 # Zd filter (mask)

""" Phi profiles """

phi_profiles_flag = 1
phi_profiles_title_flag = 1
phi_profiles_save_origin_flag = 0
phi_profiles_mode = 'mono' # mono, time_intervals, shots, ebeams

""" XY Lines Plot """
# Plot for Physic of Atomic Nuclei
xy_lines_plot_flag = 0

""" Inspect scans """

inspect_scans_flag = 0

# %% Import Data
print('\nimporting data\n')
print('slit: ', slit)
print('shot list: ', path_load_list)

# Load list of shots and energies
amount_of_shots, shots, energies, time_intervals = imd.load_shots(path_load_list)

# define result array
res = np.zeros([0, 15])

# mode = 'loader' - load signals via SigView Loader AND pickle them to a file
# mode = 'pickle' - load signals from pickled files path: objects/%shot%_file.obj or if there is not - load files
mode = 'pickle'

ne_list = []

force_reload = ["73150", "73191", "73203"]
counter_not_loaded = 0

df_inspector = pd.DataFrame()

# load signals from database
for i in range(len(shots)):
    
    print(f'\n{i+1}/{amount_of_shots}')
    
    # if str(shots[i]) in force_reload:
    #     mode = 'loader'
    
    dens = imd.load_signals(mode, 'ne', shots[i], slit, time_intervals[i], path_obj)
    alpha2 = imd.load_signals(mode, 'A2', shots[i], slit, time_intervals[i], path_obj)
    Itot = imd.load_signals(mode, 'Itot', shots[i], slit, time_intervals[i], path_obj)
    RMSItot = imd.load_signals(mode, 'RMSItot', shots[i], slit, time_intervals[i], path_obj)
    RelRMSItot = imd.load_signals(mode, 'RelRMSItot', shots[i], slit, time_intervals[i], path_obj)
    Phi = imd.load_signals(mode, 'Phi', shots[i], slit, time_intervals[i], path_obj)
    RMSPhi = imd.load_signals(mode, 'RMSPhi', shots[i], slit, time_intervals[i], path_obj)
    Zd = imd.load_signals(mode, 'Zd', shots[i], slit, time_intervals[i], path_obj)
    A_QCM = imd.load_signals(mode, 'A_QCM', shots[i], slit, time_intervals[i], path_obj)
    
    if inspect_scans_flag:
        # add A_QCM to inspector's dataframe. Plot it and inspect
        # df_buffer = pd.DataFrame()
        # df_inspector[f"{time_intervals[i]}"] = (A_QCM.x, A_QCM.y)
        pass
    
    ne_mean = np.mean(dens.y)
    ne_list.append(ne_mean)
    print('shot #{}, E={}'.format(shots[i], energies[i]))
    print('<ne> = {:.3f}'.format(ne_mean))
    print(f'time {time_intervals[i]}')
    
    # correct Phi according to line averaged density
    # dens_interp = interpolate.interp1d(dens.x, dens.y)
    # Phi.y = Phi.y * dens_interp(Phi.x) / dens_base
    
    # make Ua2 interpolants
    inds = alpha2.y.argsort()
    
    # print(len(inds), len(A_QCM.y))
    # print(alpha2.y[inds], A_QCM.y)
    
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
    
    try:
        # A_QCM.resampleas(alpha2)
        A_QCM_interp = interpolate.interp1d(bin_mean(alpha2.y[inds]),
                                          bin_mean(A_QCM.y[inds]),
                                          bounds_error=False)
    except:
        A_QCM.resampleas(alpha2)
        A_QCM_interp = interpolate.interp1d(bin_mean(alpha2.y[inds]),
                                          bin_mean(A_QCM.y[inds]),
                                          bounds_error=False)
        # print("\nNOT LOADED\n")
        # print(len(alpha2.y))
        # print(alpha2.y)
        # print(len(A_QCM.y))
        # print(A_QCM.y)
        # counter_not_loaded += 1
        pass
        # raise IndexError("A_QCM and A2 aren't equal")
    
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
                                  Zd_interp(Ua2), A_QCM_interp(Ua2)]], axis=0)

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
df['A_QCM'] = res[:,14]
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
df['A_QCM'] = pd.to_numeric(df['A_QCM'], errors='coerce')
# save copy of raw data
df_imported = copy.deepcopy(df)

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
    if intdots_flag:
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

text_size = 40

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.titlesize'] = text_size-15
plt.rcParams['axes.labelsize'] = text_size
plt.rcParams['xtick.labelsize'] = text_size
plt.rcParams['ytick.labelsize'] = text_size
plt.rcParams['legend.fontsize'] = text_size-5
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['axes.grid.axis'] = "both"
cbar_pad = 0.013 #0.1
# sns.color_palette('Dark2')
# plt.rcParams['figure.figsize'] = (5, 10)

#%% Plot Phi profiles

def plot_point(ax, point, angle, length):
     '''
     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.

     Will plot the line on a 10 x 10 plot.
     '''

     # unpack the first point
     x, y = point

     # find the end point
     endy = y + length * math.sin(math.radians(angle))
     endx = x + length * math.cos(math.radians(angle))
     ax.plot([x, endx], [y, endy], linewidth=1, color='red', alpha=0.6)

if phi_profiles_flag:
    
    if phi_profiles_mode == "angle":
        
        r, theta = cart2pol(df['x'].to_numpy(), df['y'].to_numpy())
    
        rho = df['rho'].to_list()
        
        phi = df['Phi'].to_list()
        
        df_angle = pd.DataFrame()
        df_angle['x'] = df['x'].to_numpy()
        df_angle['y'] = df['y'].to_numpy()
        df_angle['r'] = r
        df_angle['theta'] = theta * 180/np.pi
        df_angle['rho'] = rho
        df_angle['phi'] = phi
        
        angle_treshold = 1
        
        for angle in range(0, 80, 10):
            fig_angle, ax_angle = plt.subplots(1,2)
            taken_dots=[]
            plot_point(ax_angle[1], (0,0), angle, 30)
            fig_angle.canvas.manager.set_window_title(f'{angle} deg')
            for i in range(len(rho)):
                if abs(df_angle['theta'][i] - angle) <= angle_treshold:
                    point = ax_angle[0].errorbar(rho[i], 
                                                phi[i], 
                                                xerr=0.5, yerr=0.05, fmt='o',
                                                ecolor='gray', elinewidth=0.5, capsize=4, 
                                                capthick=0.7, mfc='black', mec='black',
                                                marker='o',markersize=5.)
                    
                    taken_dots.append(rho[i])
                    
                    # ax_angle.set_xlim(-5, 30)
                    ax_angle[0].set_ylim(-1.55, 0.25)
                            # label=f'Shot: {0}\nE: {energies[i]}\nTI: {0}\nne: {0}')
                    cursor = mplcursors.cursor(point, highlight=True, multiple=True)
                    
            # twoD_plot(ax_angle[1], taken_dots)
            
    if phi_profiles_mode == "mono":
        
        fig_phi_prof, ax_phi_prof = plt.subplots()
    
        x = df['rho'].to_list()
        
        y = df['A_QCM'].to_list()
        
        for i in range(len(x)):
            line = ax_phi_prof.errorbar(x, 
                                        y, 
                                        xerr=0.0, yerr=0.00, fmt='o',
                                        ecolor='gray', elinewidth=0.0, capsize=4, 
                                        capthick=0.0, mfc='black', mec='black',
                                        marker='o',markersize=5.)
                    # label=f'Shot: {0}\nE: {energies[i]}\nTI: {0}\nne: {0}')
    
        cursor = mplcursors.cursor(line, highlight=True, multiple=True)
        
    if phi_profiles_mode == "colorbar": 
        
        fig_phi_prof, ax_phi_prof = plt.subplots()
    
        x = df['rho']
        
        y = df['Phi']
        
        sc = plt.scatter(x,y,s=0,c=df['ne'])
    
        #create colorbar according to the scatter plot
        cmap = plt.cm.get_cmap('viridis')
        clb = plt.colorbar(sc, label=r'$\mathdefault{\bar{n}_e}$, $\mathdefault{10^{19}}$ $\mathdefault{м^{-3}}$')
        norm = matplotlib.colors.Normalize(vmin=min(df['ne']), vmax=max(df['ne']), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        time_color = np.array([(mapper.to_rgba(v)) for v in df['ne']])
        
        #loop over each data point to plot
        for x1, y1, color in zip(x, y, time_color):
            dots = plt.plot(x1, y1, 'o', color=color)
            plt.errorbar(x1, y1, yerr=0.05, lw=1, capsize=3, color=color)
            plt.errorbar(x1, y1, xerr=1, lw=1, capsize=3, color=color)
            
            cursor = mplcursors.cursor(dots, highlight=True, multiple=True)
        plt.set_cmap(cmap)

    # Plot parameters
        
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
    ax_phi_prof.set_ylabel(r'$A_{QCM}$, %', size=text_size)
    if phi_profiles_title_flag:
        ne_min = round(np.min(df['ne']), 3)
        ne_max = round(np.max(df['ne']), 3)
        ne_mean = round(np.mean(df['ne']), 3)
        ax_phi_prof.set_title(f'ne: avg: {ne_mean} ± {round((ne_max - ne_min)/2., 3)}; min: {ne_min}; max: {ne_max}', fontsize=text_size)
        
    if phi_profiles_save_origin_flag:
        df_phi_profiles_origin_save = pd.DataFrame()
        df_phi_profiles_origin_save['Rho'] = x
        df_phi_profiles_origin_save['Phi'] = y
        df_phi_profiles_origin_save.to_csv(path_origin_save, index=False, sep='\t')
        print(f'Saved to Origin: {path_origin_save}')

#%% Grid Plot

def plot_lines(df, ebeam, threshold, cmap='jet', ax=None):
    # Get the data for the current Ebeam value
    x = df.loc[df['Ebeam'] == ebeam]['x'].values
    y = df.loc[df['Ebeam'] == ebeam]['y'].values
    color = df.loc[df['Ebeam'] == ebeam]['Phi'].values

    # Plot the data using the plot_colormap function
    plot_colormap(x, y, color, cmap=cmap, ax=ax, threshold=threshold)

def plot_colormap(x, y, color, linewidth=10, outline=0, oc='black', cmap='jet', ax=None, cb=False, alpha=1, threshold=None):
    """
    Plot a line using a custom colormap.

    Arguments:
    - x: array-like. The x-coordinates of the data to plot.
    - y: array-like. The y-coordinates of the data to plot.
    - color: array-like. The colors to use for each point in the data.
    - cmap: string. The name of the colormap to use (default is 'jet').
    - ax: matplotlib Axes object. The axis to plot on (default is None, which creates a new axis).
    - threshold: float or None. The threshold distance between adjacent points below which they will be connected (default is None, which means all points will be plotted as separate dots).
    """

    sig, signal_title, min_val, max_val = get_signal(signal_name, df, plot_labels, language)
    
    norm = cm.colors.Normalize(vmax=max_val, vmin=min_val)

    # Normalize color values to range [0, 1]
    # norm = plt.Normalize(color.min(), color.max())

    # Create the colormap
    colormap = plt.cm.get_cmap(cmap)

    # Create a color map index based on the normalized color values
    color_index = np.array(norm(color))

    # Plot the data using a colormap
    if ax is None:
        fig, ax = plt.subplots()
    if threshold is not None:
        for i in range(len(x) - 1):
            if np.abs(x[i+1]-x[i]) < threshold and np.abs(y[i+1]-y[i]) < threshold:
                line,=ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=colormap(color_index[i]),
                        linewidth=linewidth, alpha=alpha,
                        path_effects=[path_effects.Stroke(linewidth=outline, foreground=oc), path_effects.Normal()])
                line.set_solid_capstyle('round')
    else:
        ax.scatter(x, y, c=colormap(color_index))
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add a colorbar
    # if ax is None:
    if cb:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=cbar_pad)
        cbar.set_label(signal_title)

# get colors for plotting grid
def get_color(ebeam):
    colors = []
    ebeams = []
    j = 0
    for i in range(160, 351, 10):
        colors.append(f'C{j}')
        ebeams.append(i)
        j += 1
    for n in range(len(ebeams)):
        if ebeam == ebeams[n]:
            return colors[n]

set_of_energies = sorted(list(set(energies)))
if grid_flag:
    
    # px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    # fig_grid = plt.figure(figsize=(2560*px, 1440*px), layout="constrained")
    # ax_grid = fig_grid.add_subplot()
    fig_grid, ax_grid = plt.subplots()
    imd.plot_t10_contour(ax_grid)
    
    #%% grid plot with gray pad

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

    mode = 'home'
    if mode=='home':
        dirname1 = "H:\\Другие компьютеры\\PC-NRCKI\\radref_calc_T-10\\output\\B22_I230_offax1"
        dirname2 = 'H:\\Другие компьютеры\\PC-NRCKI\\radref_calc_T-10\\output'
        # dirname2 = 'H:\\Другие компьютеры\\PC-NRCKI\\radref_calc_T-10\\output\\SGTM1'
    elif mode=='nrcki':
        dirname1 = "C:\\Progs\\Py\\Coils\\T-10_1Dbeam_NewAnalyzer_5slits\\output\\OLD\\B22_I230_offax1"
        dirname2 = "C:\\Progs\\Py\\Radreferences_T-10\\radref_calc_T-10\\output"
        # dirname2 = "C:\\Progs\\Py\\Radreferences_T-10\\radref_calc_T-10\\output\\SGTM1"
    
    traj0 = grid.import_traj(parameters, dirname2)
    grid.plot_grid(ax_grid, traj0, legend=False, marker_A2='*', marker_E=None,
                   A2=False, alpha_E=0.7, linewidth_E=7, language=language,
                   color_A2='blue', linewidth_A2=3, color_E='gray', cap_style='round')

    #%%    
    if detector_line_mode == "one_energy":
        one_energy = 300
        for flag_grid_one_energy in set_of_energies:
            if flag_grid_one_energy == one_energy:  # plot only for energy = 300
                ebeams = []
                texts = []
                grouped = df_imported.groupby('time_interval')
                for name in df['time_interval'].unique():
                    temp = grouped.get_group(name)
                    if temp['Ebeam'].unique() == one_energy:  # plot only for Ebeam = 300
                        grid_plot = ax_grid.plot(temp['x'].to_numpy(), temp['y'].to_numpy(), 
                                            linewidth=10, solid_capstyle='round', alpha=0.7,
                                            label='#{} {} keV'.format(int(temp['Shot'].unique()[0]), 
                                                                      int(temp['Ebeam'].unique()))) #choose color
    if detector_line_mode == "mono":
        for flag_grid_one_enegry in set_of_energies:
            ebeams = []
            texts = []
            grouped = df_imported.groupby('time_interval')
            for name in df['time_interval'].unique():
                temp = grouped.get_group(name)
                if flag_grid_one_enegry and int(temp['Ebeam'].unique()) == flag_grid_one_enegry: #need to delete
                    grid_plot = ax_grid.plot(temp['x'].to_numpy(), temp['y'].to_numpy(), 
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
                    
                    if grid_annotate_flag:
                        ax_grid.annotate(text=str(temp['Ebeam'].unique()[0]), xy=(x, y), 
                                    fontsize=text_size-15, rotation=-60, weight='bold', 
                                    c=get_color(temp['Ebeam'].unique()[0]), 
                                    # path_effects=[path_effects.Stroke(linewidth=1.75, foreground='black'),
                                    #    path_effects.Normal()],
                                    bbox=dict(boxstyle="square", alpha=1, pad=0, color='white'))
    elif detector_line_mode == "color":
        for flag_grid_one_enegry in set_of_energies:
            ebeams = []
            texts = []
            grouped = df_imported.groupby('time_interval')
            for name in df['time_interval'].unique():
                temp = grouped.get_group(name)
                grid_plot = ax_grid.plot(temp['x'].to_numpy(), temp['y'].to_numpy(), 
                                    linewidth=7, solid_capstyle='round', alpha=0.5,
                                    label='#{} {} keV'.format(int(temp['Shot'].unique()[0]), 
                                                              int(temp['Ebeam'].unique()))) #choose color
                texts.append(grid_plot)
        # annotation
            grouped = df_imported.groupby('Ebeam')
            #pos = [-0.4,-0.5,-0.5,-0.0,-0.7,-0.5,-0.3,0,0,0.2,0.3,-0.1,-0.2,5]
            for ebeam in df['Ebeam'].unique():
                
                temp = grouped.get_group(ebeam)
                
                x = temp['x'].to_numpy()[np.argmin(temp['y'].to_numpy())] - 0 #+ pos[j]
                y = np.min(temp['y'].to_numpy()) - 3
                
                if grid_annotate_flag:
                    ax_grid.annotate(text=str(temp['Ebeam'].unique()[0]), xy=(x, y), 
                                fontsize=text_size-5, rotation=-60, weight='bold',
                                path_effects=[path_effects.Stroke(linewidth=1.75, foreground='black'),
                                   path_effects.Normal()],
                                bbox=dict(boxstyle="square", alpha=1, pad=0, color='white'))
                
    elif detector_line_mode == "rainbow2":
        
        # Loop over unique Ebeam values and plot the lines separately for each Ebeam
        for i, ebeam in enumerate(df['Ebeam'].unique()):
            plot_lines(df, ebeam, grid_rainbow_threshold, cmap='jet', ax=ax_grid)
            
    elif detector_line_mode == "rainbow":
        
        plot_colormap(df['x'].to_numpy(), df['y'].to_numpy(), df['Phi'].to_numpy(),
                      cmap=cmap_2d_map, ax=ax_grid, cb=True, threshold=grid_rainbow_threshold)
    
    ax_grid.grid()
    ax_grid.set_aspect('equal', adjustable='box')
    ax_grid.set_xlabel(plot_labels[language]['xlabel'], size=text_size)
    ax_grid.set_ylabel(plot_labels[language]['ylabel'], size=text_size)
    ax_grid.tick_params(axis='both', labelsize=text_size)
    ax_grid.set_xlim(0, 33)
    ax_grid.set_ylim(-10, 30)
    plt.show()

#%% Plot x, y
if xy_lines_plot_flag:
    df[df.Ebeam != 240].x.to_numpy(), df[df.Ebeam != 240].y.to_numpy()
    
    time_interval_ax = 'from672.68to709.79'
    
    x_xy = df[df.time_interval == time_interval_ax].x.to_numpy()
    y_xy = df[df.time_interval == time_interval_ax].y.to_numpy()
    phi_xy = df[df.time_interval == time_interval_ax].Phi.to_numpy()
    rho_xy = df[df.time_interval == time_interval_ax].rho.to_numpy()
    t_xy = np.linspace(672.68, 709.79, len(phi_xy))
    
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig_xy = plt.figure(figsize=(1600*px, 900*px), layout="constrained")
    
    widths = [1, 1, 3]
    gs = GridSpec(3, 3, figure=fig_xy, width_ratios=widths)
    marker = 'o'
    linewidth = 4
    markersize = 10
    
    ax_xy1 = fig_xy.add_subplot(gs[0, :-1])
    ax_xy2 = fig_xy.add_subplot(gs[1, :-1])
    ax_xy3 = fig_xy.add_subplot(gs[2, :-1])
    
    # ax_map = fig_xy.add_subplot(gs[:, 2])
    
    mplcursors.cursor(ax_xy1).connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]:.2f}, {sel.target[1]:.2f})'))
    mplcursors.cursor(ax_xy2).connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]:.2f}, {sel.target[1]:.2f})'))
    mplcursors.cursor(ax_xy3).connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]:.2f}, {sel.target[1]:.2f})'))

#%% Inscpect scans ("Inspector" version=0.1)

if inspect_scans_flag:

    fig_inspector, ax_inspector = plt.subplots()
    
    for ti in time_intervals.set():
        ax_inspector.plot(df["A_QCM"])

#%% Interpolate dots and plot 2d map

if twoD_plot_flag:
    fig_map, ax_map = plt.subplots()
    
    imd.plot_t10_contour(ax_map)
    
    sig, signal_title, min_val, max_val  = get_signal(signal_name, df, plot_labels, language)
    dots_color = sig #df['Ebeam'].to_numpy()
    # min_val = 180
    # max_val = 330
    
    cmap = plt.cm.get_cmap(cmap_2d_map)
    
    if log_colorbar_flag == 0:
        norm = cm.colors.Normalize(vmax=max_val, vmin=min_val)
    elif log_colorbar_flag == 1:
        norm = cm.colors.LogNorm(vmin=0.01, vmax=0.07)
    
    x, y = np.linspace(xmin, xmax, Npoints), np.linspace(ymin, ymax, Npoints)
    x, y = np.meshgrid(x, y)
    
    if polar_coordinates_flag:
        grid = interpolate.griddata(cart2pol(df['x'].to_numpy(), df['y'].to_numpy()), 
                                    sig, cart2pol(x,y), method=interpolation_method, rescale=rescale)
    else:
        grid = interpolate.griddata((df['x'].to_numpy(), df['y'].to_numpy()), 
                                    sig, (x,y), method=interpolation_method, rescale=rescale)
    
    
    grid2 = copy.deepcopy(grid)
    
        # len_grid_x = len(grid2[0])
        # len_grid_y = len(grid2[1])
        
        # for i in range(len_grid_x):
        #     for j in range(len_grid_y):
        #         if i <= 0.45*j:
        #             grid2[i][j] = np.nan
    
    
    # mask = ~((grid[0] < 10) & (grid[0] > -10) & (grid[1] < 10) & (grid[1] > -10))
    # grid = np.c_[grid[0][mask], grid[1][mask]]
        
        
    # grid = grid2
    
    # grid = gaussian_filter(grid, 0.5)
    # Resample your data grid by a factor of 3 using cubic spline interpolation.
    # grid = scipy.ndimage.zoom(grid, 3)
    
    #Equipotential lines
    if equipotential_lines_flag:
        contour_levels = np.linspace(sig.min(), sig.max(), amount_of_lines)
        # contour_levels = [-0.442, -0.28,  -0.188, -0.061,
        #         0.056,  0.21157913,  0.33198138]
        # contour_levels = [-0.4, -0.39, -0.27, -0.15, -0.032, 0.28]
        if contourf_flag:
            contours = plt.contourf(x, y, grid, levels=contour_levels, cmap="jet")
        else:
            contours = plt.contour(x, y, grid, levels=contour_levels, colors='black', linestyles='--')
    
    if show_grid_lines_flag:
          plot_colormap(df['x'].to_numpy(), df['y'].to_numpy(), df['Phi'].to_numpy(),
                        cmap=cmap_2d_map, ax=ax_map, threshold=grid_rainbow_threshold,
                        alpha=1)
    
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
        str_A_QCM = str(round(df['A_QCM'].to_numpy()[i], 3))
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
        elif signal_name == 'A_QCM':
            label = f'Shot: {str_shot}\nE: {str_Ebeam}\nTI: {str_TI}\nne: {str_ne}\n\
    A_QCM: {str_A_QCM}\nx = {str_x}\ny = {str_y}'
        labels.append(label)
    
    if log_colorbar_flag:
        sc = ax_map.scatter(df['x'].to_numpy(), df['y'].to_numpy(), c=dots_color, 
                        norm=norm, s=100*show_dots_flag, cmap=cmap, edgecolors='black')
    else:
        sc = ax_map.scatter(df['x'].to_numpy(), df['y'].to_numpy(), c=dots_color, 
                        vmin=min_val, vmax=max_val, s=100*show_dots_flag, 
                        cmap=cmap, edgecolors='black')
        sc = ax_map.scatter(df['x'].to_numpy(), df['y'].to_numpy(), c=dots_color, 
                        vmin=min_val, vmax=max_val, s=250*show_dots_flag, 
                        cmap=cmap, edgecolors='black')
        # sc = ax_map.scatter(df[df.Ebeam != 240].x.to_numpy(), df[df.Ebeam != 240].y.to_numpy(), c='gray', 
        #                 vmin=min_val, vmax=max_val, s=250*show_dots_flag, 
        #                 cmap=cmap, edgecolors='black')
            
    mplcursors.cursor(ax_map).connect(
        "add", lambda sel: sel.annotation.set_text(labels[sel.index]))
    
    if interpolation_flag:
        ax_map.imshow(grid, interpolation=imshow_interpolation,
                    origin='lower',
                    extent=[xmin, xmax, ymin, ymax],
                    norm=norm, cmap=cmap)
    
    if colorbar_flag:
        bounds= np.arange(-0.5, 0.5, 0.1)
        cbar = plt.colorbar(sc, ax=ax_map, pad=cbar_pad)#, boundaries=bounds, ticks=bounds)
        cbar.set_label(label=signal_title, size=text_size)
        cbar.ax.tick_params(labelsize=text_size)
    
    str_ne_mean = str(round(ne_mean, 3))
    if show_title_flag:
        polar_title = 'polar' if polar_coordinates_flag else 'cartesian'
        title = str(len(df)) + f' dots; ne: {str_ne_mean} ± {ne_eps}, samples={Npoints}, {polar_title}, {interpolation_method}'
        ax_map.set_title(title)

    ax_map.grid()
    ax_map.set_aspect('equal', adjustable='box')
    ax_map.set_xlabel(plot_labels[language]['xlabel'], size=text_size)
    ax_map.set_ylabel(plot_labels[language]['ylabel'], size=text_size)
    ax_map.tick_params(axis='both', labelsize=text_size)
    ax_map.set_xlim(1.17, 28.93)
    ax_map.set_ylim(-4.76, 22.93)
    plt.show()
    
if xy_lines_plot_flag:
    #XY plot
    # ax_map.set_xlim(1.17, 28.93)
    # ax_map.set_ylim(-4.76, 22.93)
    
    linewidth = 1
    size = 300
    color='blue'
    
    ax_xy1.plot(t_xy, x_xy, linewidth=5, zorder=1)
    ax_xy2.plot(t_xy, y_xy, linewidth=5, zorder=1)
    ax_xy3.plot(t_xy, phi_xy, linewidth=5, zorder=1)
    
    ax_xy1.scatter(t_xy, x_xy, marker=marker, linewidth=linewidth, c=phi_xy, 
                   cmap=cmap, norm=norm, edgecolors='black', linewidths=linewidth, s=size, zorder=2)
    
    ax_xy2.scatter(t_xy, y_xy, marker=marker, linewidth=linewidth, c=phi_xy, 
                   cmap=cmap, norm=norm, edgecolors='black', linewidths=linewidth, s=size, zorder=2)
    
    ax_xy3.scatter(t_xy, phi_xy, marker=marker, linewidth=linewidth, c=phi_xy, 
                   cmap=cmap, norm=norm, edgecolors='black', linewidths=linewidth, s=size, zorder=2)

    ax_xy1.grid()
    ax_xy2.grid()
    ax_xy3.grid()

    ax_xy3.set_xlabel('t, с', size=text_size)
    ax_xy1.set_ylabel(plot_labels[language]['xlabel'], size=text_size)
    ax_xy2.set_ylabel(plot_labels[language]['ylabel'], size=text_size)
    ax_xy3.set_ylabel('sig, у.е.', size=text_size)
    ax_xy1.tick_params(axis='both', labelsize=text_size)
    ax_xy2.tick_params(axis='both', labelsize=text_size)
    ax_xy3.tick_params(axis='both', labelsize=text_size)

    ax_xy1.set_xlim(672.68, 709.79)
    ax_xy2.set_xlim(672.68, 709.79)
    ax_xy3.set_xlim(672.68, 709.79)
    
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    # plt.tight_layout()
    plt.show()