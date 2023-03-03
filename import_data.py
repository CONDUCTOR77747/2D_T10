# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:00:57 2022

@author: NRCKI-261
"""
import numpy as np
import pickle
from loadsig import sendLoaderCmd
from os.path import exists
from hibpsig import signal
#%% Data and pathes

mode = "OH"

if mode == "allec":
    # B22_I230_ABC (allec)
    #1 CreateList
    path_CreateList_save_lists = 'Lists\\'
    path_CreateList_save_2dmaps_lists = 'lists\\2dmaps\\T10_B22_I230\\allec\\allec.txt'
    
    #2 2d_t10
    path_2d_t10_load = path_CreateList_save_2dmaps_lists
    path_2d_t10_save = 'lists\\2dmaps\\T10_B22_I230\\allec\\sorted\\allec.txt'
    path_2d_t10_save_obj = 'objects\\T10_B22_I230\\allec\\'
    
    #3 Phi_profiles
    path_phi_profiles_list = path_2d_t10_save
    path_phi_profiles_list_outside_the_polygon = 'lists\\2dmaps\\T10_B22_I230\\allec\\sorted\\allec_otp.txt'
    path_phi_profiles_save_obj = 'objects\\T10_B22_I230\\allec\\Phi_profiles\\'
    path_phi_profiles_save_origin = 'lists\\2dmaps\\T10_B22_I230\\allec\\phi_profiles.dat'
    path_phi_profiles_save_multicarpet = 'lists\\2dmaps\\T10_B22_I230\\allec\\'
    
    #4 Plot_carpets
    path_plot_carpets_list = path_phi_profiles_list
    path_plot_carpets_save_obj = "objects\\T10_B22_I230\\allec\\Plot_carpets\\"
    path_plot_carpets_save_img = "lists\\2dmaps\\T10_B22_I230\\allec\\carpets\\"

elif mode == "offax":
    # B22_I230_AC (offax)
    #1 CreateList
    path_CreateList_save_lists = 'Lists\\'
    path_CreateList_save_2dmaps_lists = 'lists\\2dmaps\\T10_B22_I230\\offax\\offax.txt'
    
    #2 2d_t10
    path_2d_t10_load = path_CreateList_save_2dmaps_lists
    path_2d_t10_save = 'lists\\2dmaps\\T10_B22_I230\\offax\\sorted\\offax.txt'
    path_2d_t10_save_obj = 'objects\\T10_B22_I230\\offax\\'
    
    #3 Phi_profiles
    path_phi_profiles_list = path_2d_t10_save
    path_phi_profiles_list_outside_the_polygon = 'lists\\2dmaps\\T10_B22_I230\\offax\\sorted\\offax_otp.txt'
    path_phi_profiles_save_obj = 'objects\\T10_B22_I230\\offax\\Phi_profiles\\'
    path_phi_profiles_save_origin = 'lists\\2dmaps\\T10_B22_I230\\offax\\phi_profiles.dat'
    path_phi_profiles_save_multicarpet = 'lists\\2dmaps\\T10_B22_I230\\offax\\'
    
    #4 Plot_carpets
    path_plot_carpets_list = path_phi_profiles_list
    path_plot_carpets_save_obj = "objects\\T10_B22_I230\\offax\\Plot_carpets\\"
    path_plot_carpets_save_img = "lists\\2dmaps\\T10_B22_I230\\offax\\carpets\\"
    
elif mode == "OH":
    # B22_I230_OH
    #1 CreateList
    path_CreateList_save_lists = 'Lists\\'
    path_CreateList_save_2dmaps_lists = 'lists\\2dmaps\\T10_B22_I230\\OH\\OH.txt'
    
    #2 2d_t10
    path_2d_t10_load = path_CreateList_save_2dmaps_lists
    path_2d_t10_save = 'lists\\2dmaps\\T10_B22_I230\\OH\\sorted\\OH.txt'
    path_2d_t10_save_obj = 'objects\\T10_B22_I230\\OH\\'
    
    #3 Phi_profiles
    path_phi_profiles_list = path_2d_t10_save
    path_phi_profiles_list_outside_the_polygon = 'lists\\2dmaps\\T10_B22_I230\\OH\\sorted\\OH_otp.txt'
    path_phi_profiles_save_obj = 'objects\\T10_B22_I230\\OH\\Phi_profiles\\'
    path_phi_profiles_save_origin = 'lists\\2dmaps\\T10_B22_I230\\OH\\phi_profiles.dat'
    path_phi_profiles_save_multicarpet = 'lists\\2dmaps\\T10_B22_I230\\OH\\'
    
    #4 Plot_carpets
    path_plot_carpets_list = path_phi_profiles_list
    path_plot_carpets_save_obj = "objects\\T10_B22_I230\\OH\\Plot_carpets\\"
    path_plot_carpets_save_img = "lists\\2dmaps\\T10_B22_I230\\OH\\carpets\\"

# B22_I230 load radrefs
def load_radrefs(shot, slit, ebeam, mode='signal'):
    """
    Provides path to radref according to shot number.

    Parameters
    ----------
    shot : int
        Shot number.
    slit : string
        Slit number.
    ebeam : int
        Beam energy.
    mode : string, optional
        'signal' - SigView format with %Ebeam%.
        'file' - Direct path to file with energy.
        The default is 'signal'.

    Returns
    -------
    String
        Path to radref file (File mode) or signal name 
        for SigView (Signal mode).

    """
    radref_fname = 'C:\\Progs\\radrefs\\'
    
    if mode == 'signal':
        radref_name = 'toPresetE%Ebeam%_' + str(slit) + '.dat:0:4'
    elif mode == 'file':
        radref_name = 'toPresetE' + str(ebeam) + '_' + str(slit) + '.dat'
        
    path = radref_fname + 'B22_I230_offax\\' + radref_name
    path1 = radref_fname + 'B22_I230_allEC\\' + radref_name
    path2 = radref_fname + '73214_mode_3\\' + radref_name
    path3 = radref_fname + '73146_mode_3\\' + radref_name
    
    if (73127<=shot<=73137) or (73186<=shot<=73189) or (73199<=shot<=73205):
        return path
    elif ((73138<=shot<=73152) or (73190<=shot<=73199)) and shot != 73146:
        return path1
    elif shot == 73214:
        return path2
    elif shot == 73146:
        return path3
#%% operation
DAS_FILES_path = 'C:\\Progs\\DAS_FILES\\'
slit = 'slit3'
# %% configure LOADER
sendLoaderCmd(device='t10', mode='mmap', dtype='float32')
sendLoaderCmd(show_server=False)
#%% functions

# get signal names for loading via signal function (Reonid Loader)
def get_signame(name, slit, time_interval, radref=''):
    """
    Provides signal names in SigViewer format.

    Parameters
    ----------
    name : string
        Signal name. Use: ne, Itot, Phi, Zd, ECRH, Radius, Ipl, A2, RMSPhi,
        RMSItot, RelRMSItot.
    slit : string
        Slit number. Exxample: 'slit3'.
    time_interval : string
        Time intervals for signal
        'fromXXX.XXtoXXX.XX'.
    radref : string, optional
        Path to radref. Only for Radius signal. The default is ''.

    Returns
    -------
    string
        Signal name for SigViewer (@Reonid Progs).
        Example: 'T10HIBP::Itot{' + slit + ', avg333n11,, rar22, \
                     clean, noz, ' + time_interval + '}''.

    """
    if name == 'ne':
        return 'I.f8{x0.1333, z200, avg33,' + time_interval + '}'
    
    elif name == 'Itot':
        return 'T10HIBP::Itot{' + slit + ', avg333n11,, rar22, \
                     clean, noz, ' + time_interval + '}'
    
    elif name == 'Phi':
        return 'T10HIBP::Phi{' + slit + \
                     ', clean, noz, rar22, avg1111n11, G=2.8569, ' \
                     + time_interval + '}'
                     
    elif name == 'Zd':
         return 'T10HIBP::Zd{' + slit + ', avg333n11, rar22, brk, clean, noz, ' + \
                       time_interval + '}'

    elif name == 'ECRH':
         return 'I.EC{x-1, z80, avg33}'
     
    elif name == 'Radius':
         return 'T10HIBP::Radius{noz, avg1111, rar22, clean, \
     ?' + radref + ',' + time_interval + '}'
     
    elif name == 'Ipl':
         return 'I.I{avg33}'
     
    elif name == 'A2':
         return 'T10HIBP::A2{' + slit + ', avg111, rar22, ' +\
                         time_interval + '}'
     
    elif name == 'RMSPhi':
         return 'T10HIBP::Phi{' + slit + ', rms1001, clean, noz, \
                         rar22, ' + time_interval + '}'
     
    elif name == 'RMSItot':
         return 'T10HIBP::Itot{' + slit + ', rms1001, rar22, \
                          clean, noz, ' + time_interval + '}'
     
    elif name == 'RelRMSItot':
         return 'T10HIBP::Itot{' + slit + ', relrms1001, rar22,\
                             clean, noz, ' + time_interval + '}'

def pickle_obj(mode, path, sig=''):
    """
    Creates obj files for faster signal loading.

    Parameters
    ----------
    mode : string
        'save' - save signal to obj file.
        'load' - load signal from obj file.
    path : string
        Save/Load path.
    sig : Signal Class (hibpsig), optional
        Signal to save. Only for save mode. The default is ''.

    Returns
    -------
    sig : Signal Class (hibpsig)
        Signal imported with loader from SigView.

    """
    if mode == 'save':
        signal_pickle = open(path, 'wb')
        pickle.dump(sig, signal_pickle)
        signal_pickle.close()
    elif mode == 'load':
        signal_pickle = open(path, 'rb')
        sig = pickle.load(signal_pickle)
        signal_pickle.close()
        return sig

def load_signals(mode, name, shot, slit, time_interval, path='', radref=''):
    """
    Load signals from SigView via Loader (@Reonid progs).

    Parameters
    ----------
    mode : string
        'loader' - load signal AND save to obj file.
        'pickle' - load signal FROM obj file.
        'no save' - load signal WITH OUT save to obj file.
    name : string
        ne, Itot, Phi, Zd, ECRH, Radius, Ipl, A2, RMSPhi,
       RMSItot, RelRMSItot.
    shot : int
        Shot number.
    slit : string
        Slit number.
    time_interval : string
        Time interval (diapason) of signal.
        Example: 'fromXXX.XXtoXXX.XX'
    path : string, optional
        Path to save obj files. The default is ''.
        Not for 'no save' mode.
    radref : string, optional
        Only for Radius. The default is ''.

    Returns
    -------
    sig : Signal Class
        Signal imported with loader from SigView.

    """
    fname = path + str(shot) + '_' + time_interval + '_' + name + '.obj'
    if mode == 'loader': 
        sig = signal(shot, get_signame(name, slit, time_interval, radref))
        pickle_obj('save', fname, sig)
        return sig
    elif mode == 'pickle':
        if exists(fname):
            sig = pickle_obj('load', fname)
        else:
            sig = signal(shot, get_signame(name, slit, time_interval, radref))
            pickle_obj('save', fname, sig)
        return sig
    elif mode == 'no save':
        sig = signal(shot, get_signame(name, slit, time_interval, radref))
        return sig

# load shot numbers, energies and time intervals from file
def load_shots(filename):
    """
    Reads shot numbers, Ebeams, signal time intervals from txt file.

    Parameters
    ----------
    filename : string
        Path to txt file. File consists of rows: "shot !ebeam !time interval".

    Returns
    -------
    amount_of_shots : int
        Amount of loaded shots.
    shots : list of str
        Shot number. Example: ['73150', '73151'].
    energies : list of str
        Ebeam for each shot. Example: ['180', '190'].
    time_intervals : list of str
        Time intervals for signal.
        'fromXXX.XXtoXXX.XX'.

    """
    shots = []
    energies = []
    time_intervals = []
    with open(filename) as f:
        for line in f.readlines():
            li = line.split('!')
            # print(li)
            shots.append(int(li[0]))
            energies.append(int(li[1]))
            time_intervals.append(str(li[2]).rstrip())
    f.close()
    amount_of_shots = len(shots)
    return amount_of_shots, shots, energies, time_intervals
#%% define T-10 contour
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