# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:52:53 2023

@author: ammosov_yam
"""

# -*- coding: utf-8 -*-
# Import traj coordinates from dat-file
#file "traj****.dat" should contain the following
#line0 z_last Ua2 E B0
#line1 x y z rho tag Bx By Bz
#line2 ... 
import numpy as np
import math
import matplotlib.pyplot as plt
from os import path
import os
import re
import send2trash

#%% import trajectories

def import_traj(parameters, dirname, print_log=False):
    
    #parameters setup
    
    #numbers of trajects
    N_start = parameters['N_start']
    N_stop = parameters['N_stop']
    #Beam energy
    E_start = parameters['E_start']
    E_stop = parameters['E_stop']
    E_step = parameters['E_step']
    #Slit numbers
    slit_start = parameters['slit_start']
    slit_stop = parameters['slit_stop']
    #Number of thin filaments
    n_filaments = parameters['n_filaments']
    n_dim = parameters['n_dim']

    size = int(n_filaments*(N_stop-N_start+1)*(slit_stop-slit_start+1)*((E_stop-E_start)/E_step+1))
    
    traj0 = np.full(size,np.nan, 
                    dtype=[('E','float32'), 
                           ('B0','float32'),
                           ('Ua2','float32'),
                           ('No','float32'),
                           ('fil','float32'),
                           ('slit','float32'),
                           ('z_last','float32'),
                           ('rho_ion','float32'),
                           ('x_ion','float32'),
                           ('y_ion','float32'),
                           ('z_ion','float32'),
                           ('lam','float32'),
                           ('Ipl','float32'),
                           ('rho', '({},)float32'.format(n_dim)),
                           ('x', '({},)float32'.format(n_dim)),
                           ('y', '({},)float32'.format(n_dim)),
                           ('z', '({},)float32'.format(n_dim)),
                           ('tag', '({},)float32'.format(n_dim))])
                           # ('Bx', '({},)float32'.format(n_dim)),
                           # ('By', '({},)float32'.format(n_dim)),
                           # ('Bz', '({},)float32'.format(n_dim))])
                           
    traj0 = np.rec.array(traj0)
    k = 0
    
    #import
    
    if print_log: print("\n***Importing ...***")
    for i_E in np.arange(E_start,E_stop+1,E_step):
        Ebeam = str(i_E)
        if print_log: print('\n E = {} \n'.format(Ebeam))
        for i in range(N_start,N_stop+1):
            traj_number = str(i)
            for l in range(slit_start,slit_stop+1):
                slit_number = str(l)
                for m in range(1,n_filaments+1):
                    if m < 10:
                        filament_number = '0' + str(m)
                    else:
                        filament_number = str(m)
                    
                    filename = path.join(dirname, 'E' + Ebeam + '_traj' + traj_number +
                                         '_slit' + slit_number + '_fil' + filament_number + '.dat')
                    try:
                        with open(filename) as f:
                            traj0[k].slit = float(slit_number)
                            traj0[k].No = float(traj_number)
                            line0 = f.readline()
                            line0 = line0.split()
                            traj0[k].z_last = line0[0]
                            traj0[k].Ua2 = line0[1]
                            traj0[k].E = line0[2]
                            traj0[k].B0 = line0[3]
                            traj0[k].Ipl = line0[4]
                            traj0[k].fil = m
                            rzone = []
                            xzone = []
                            yzone = []
                            zzone = []
                            other_lines = f.readlines()
                            for j in range(len(other_lines)):
                                line1 = other_lines[j].split()
                                traj0[k].x[j] = float(line1[0]) / 100 # [m]
                                traj0[k].y[j] = float(line1[1]) / 100 # [m]
                                traj0[k].z[j] = float(line1[2]) / 100 # [m]
                                traj0[k].rho[j] = float(line1[3]) / 100 # [m]
                                # teraj[k].rho[j] = ret_rho(traj[k].x[j],traj[k].y[j],traj[k].z[j])
                                traj0[k].tag[j] = float(line1[4])
                                # traj0[k].Bx[j] = float(line1[5])
                                # traj0[k].By[j] = float(line1[6])
                                # traj0[k].Bz[j] = float(line1[7])
                                
                                if traj0[k].tag[j] == 0 and traj0[k].rho[j] < 0.4:
                                    rzone = np.append(rzone,traj0[k].rho[j])
                                    xzone = np.append(xzone,traj0[k].x[j])
                                    yzone = np.append(yzone,traj0[k].y[j])
                                    zzone = np.append(zzone,traj0[k].z[j])
                            traj0[k].rho_ion = 0.5*(rzone[0]+rzone[-1]) #rzone[round(len(rzone)/2)]
                            traj0[k].x_ion = 0.5*(xzone[0]+xzone[-1]) #xzone[round(len(xzone)/2)]
                            traj0[k].y_ion = 0.5*(yzone[0]+yzone[-1]) #yzone[round(len(yzone)/2)]
                            traj0[k].z_ion = 0.5*(zzone[0]+zzone[-1]) #zzone[round(len(zzone)/2)]
                            # calculate lambda for a single filament
                            traj0[k].lam = math.sqrt((xzone[-1]-xzone[0])**2+
                                                    (yzone[-1]-yzone[0])**2+
                                                    (zzone[-1]-zzone[0])**2)
                            k = k+1
                    except FileNotFoundError:
                        if print_log: print("E{} traj {} slit {} filament {} NOT FOUND".format(i_E,i,l,m))
                        pass
    
    # delete all NaNs from traj
    mask = ~np.isnan(traj0.No)
    traj0 = traj0[mask]
    
    if print_log: print("\n*** {} trajectories imported ***".format(k))
    if print_log: print(parameters['dirname'])   
    if print_log: print("\n*** Work complete ***")
    
    return traj0
#%% plot t10 tokamak contour

def plot_t10_contour(ax):
    # define T-10 contour
    camera = np.array([[0,101.200000000000],
        [27.6000000000000,101.200000000000],
        [33.2000000000000,101.200000000000],
        [36.4000000000000,101.200000000000],
        [36.4000000000000,36],
        [44,36],
        [78,56],
        [82,57.8000000000000],
        [86.8000000000000,68.6000000000000],
        [85.2000000000000,69.2000000000000],
        [87.6000000000000,74.8000000000000],
        [96.4000000000000,71],
        [94,65.4000000000000],
        [92.4000000000000,66],
        [90,60],
        [91.6000000000000,60.6000000000000],
        [103.600000000000,67.4000000000000],
        [100.800000000000,72.2000000000000],
        [103.600000000000,73.8000000000000],
        [116.800000000000,50.6000000000000],
        [114,49.2000000000000],
        [111,54],
        [99.8000000000000,48],
        [87.8000000000000,40],
        [80.8000000000000,36],
        [105,36],
        [105,0]])

    #plot camera and plasma contour
    camera = camera/100 # [m]
    ax.plot(camera[:,0]*100,camera[:,1]*100)
    a = 0.3*100 #plasma radius [cm]
    Del = -0.01 #plasma shift [m]
    plasma_contour = plt.Circle((Del, 0), a, color='r',linewidth = 1.5, fill=False)
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
        # surf = np.append(surf,[[i+Del,0]],axis = 0)
        surf = np.append(surf,[[Del,0]],axis = 0)
        if 100*i % 5 == 0:
            ax.plot(surf[:,0]*100 ,surf[:,1]*100, linewidth=2, color='k', alpha=1)
        else:
            ax.plot(surf[:,0]*100 ,surf[:,1]*100, linewidth=1, color='k', alpha=1)
            
# %% plot grid
def plot_grid(ax, traj0, slit=3, fil=3, marker_A2 = '*', marker_E = '*',
              color_A2 = 'blue', color_E = 'green', textsize = 40, linewidth=3,
              axes='XY', language='ru', A2=True, E=True, grid=True, legend=True,
              ax_xz=None, A2_xz=False, E_xz=False, print_minmaxes=False,
              linewidth_E=5, linewidth_A2=5, cap_style=None,
              xlim=[1, 33], ylim=[-12, 21], alpha_A2=1, alpha_E=1):
    
    labels = {
        'ru':  {'xlabel': 'x, см', 'ylabel_y': 'y, см', 'ylabel_z': 'z, см', 'kV': 'кВ', 'keV':'кэВ'},
        'eng': {'xlabel': 'x, cm', 'ylabel_y': 'y, cm', 'ylabel_z': 'z, cm', 'kV': 'kV', 'keV':'keV'}
             }
    
    if ax_xz is not None:
        A2_xz, E_xz = True, True
    
    mask = (traj0.slit == slit)&(traj0.fil == fil)
    traj = traj0[mask]
    
    E_vals = np.unique(traj.E)
    N_E = E_vals.shape[0]
    UA2_vals = np.unique(traj.Ua2)
    N_A2 = UA2_vals.shape[0]
    
    E_grid = np.zeros((N_A2,3,N_E))
    E_grid[:] = np.nan
    A2_grid = np.zeros((N_E,3,N_A2))
    A2_grid[:] = np.nan
    
    if print_minmaxes:
        for i, E in enumerate(E_vals):
            A2_vals = traj.Ua2[traj.E == E]
            A2_min = np.min(A2_vals)
            A2_max = np.max(A2_vals)
            print(f'{E}: {A2_min} - {A2_max} ({len(A2_vals)})\n')
    
    linestyle_E = '-'
    linestyle_A2 = '-'
    
    #make a grid of constant E
    for i_E in range(0,N_E,1):
        k = -1
        for i in range(len(traj)):
            if  traj[i].E == E_vals[i_E]:
                k += 1
                E_grid[k,:,i_E] = [traj[i].x_ion, traj[i].y_ion, traj[i].z_ion]
        
        if E:
            line_E, = ax.plot(E_grid[:,0,i_E]*100, E_grid[:,1,i_E]*100,
                     linestyle=linestyle_E, color=color_E, linewidth=linewidth_E,
                     marker=marker_E, alpha=alpha_E,
                     label=str(int(E_vals[i_E]))+' '+labels[language]['keV'])
            
            if cap_style == 'round':
                line_E.set_solid_capstyle('round')

            if E_xz:
                ax_xz.plot(E_grid[:,0,i_E]*100, E_grid[:,2,i_E]*100,
                        linestyle=linestyle_E, linewidth=linewidth_E,
                        marker=marker_E, alpha=alpha_E,
                        label=str(int(E_vals[i_E]))+' '+labels[language]['keV'])
    
    #make a grid of constant A2
    for i_A2 in range(0,N_A2,1):
        k = -1
        for i in range(len(traj)):
            if traj[i].Ua2 == UA2_vals[i_A2]:
                k += 1
                A2_grid[k,:,i_A2] = [traj[i].x_ion, traj[i].y_ion, traj[i].z_ion]
    
        if A2:
            ax.plot(A2_grid[:,0,i_A2]*100, A2_grid[:,1,i_A2]*100,
                  linestyle=linestyle_A2, linewidth=linewidth_A2,  color=color_A2,
                  marker=marker_A2, alpha=alpha_A2,
                  label=str(round(UA2_vals[i_A2],1))+' '+labels[language]['kV'])
            
            if A2_xz:
                ax_xz.plot(A2_grid[:,0,i_A2]*100, A2_grid[:,2,i_A2]*100,
                    linestyle=linestyle_A2, linewidth=linewidth_A2,  color=color_A2,
                    marker=marker_A2, alpha=alpha_A2,
                    label=str(round(UA2_vals[i_A2],1))+' '+labels[language]['kV'])
    
    # Plotting Params
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['axes.titlesize'] = textsize-27
    plt.rcParams['axes.labelsize'] = textsize
    plt.rcParams['xtick.labelsize'] = textsize
    plt.rcParams['ytick.labelsize'] = textsize
    plt.rcParams['legend.fontsize'] = textsize-27
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['axes.grid.axis'] = "both"
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(labels[language]['xlabel'], size=textsize)
    ax.set_ylabel(labels[language]['ylabel_y'], size=textsize)
    ax.tick_params(axis='both', labelsize=textsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if grid: ax.grid()
    if legend: ax.legend()

    plt.show()


def delete_files_with_number(dirname, num):
    pattern = fr'.*E{num:03d}.*(\.dat|\.DAT)$'
    for entry in os.scandir(dirname):
        if entry.is_file() and re.match(pattern, entry.name):
            send2trash.send2trash(entry.path)
            print(f"Deleted file: {entry.name}")
