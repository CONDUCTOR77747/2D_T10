# -*- coding: utf-8 -*-
"""
Load Signal with help of SignalLoader server (sigload.exe)
SignalLoader should be started manually only once, 
after that it will be found automatically. 

Signal names are fully compatible with SigViewer
  including options  {avg101, x1.5, rar10, ...}
            prefixes HIBP::, FILE:: etc
            substitutions %SHOT%, %TJIIEBEAMII% etc

@author: reonid


Exports:  
    WinRegKey 
    loadSignal(shot, sig, **kwargs)
    
"""

import struct
import ctypes
import inspect

import win32api, win32con, win32gui
import winreg 
import subprocess
import time
import mmap

import numpy as np
import matplotlib.pylab as plt

class SignalError(Exception): pass
class SignalLoaderError(Exception): pass
class SignalsAreNotCompatible(Exception): pass 
    
TYPE_DOUBLE = 3
TYPE_SINGLE = 5
TYPE_INTEGER = 8
TYPE_SMALLINT = 11

SIG_MMAP_NAME = 'REONID@SIGNALDATA'
SIG_MMAP_SIGNATURE = 0x010AD516
WM_USER = 0x0400
WM_SIGLOADER_CMD = WM_USER + 247
WM_MY_SERVER_PING = WM_USER + 1973
SIGNAME_MAXLEN = 512

CMD_CHANGE_DEVICE = 22
CMD_CHANGE_TYPEID = 33
CMD_CHANGE_TRANSPORT_MODE = 44
CMD_CLOSE_MMAP = 55
CMD_SHOW_SERVER = 77

DEVICE_TJII = 1002
DEVICE_T10 = 1010
DEVICE_COMPASS = 1014
DEVICE_LIVEN = 1017

#---------------------------- Windows utilities -------------------------------

class WinRegKey: 
    def __init__(self, hive, key_name): 
        self.hive = hive
        self.key_name = key_name

    def __enter__(self):
        self.key_handle = winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.key_name)
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        winreg.CloseKey(self.key_handle)
        if exception_type: 
            raise 
    
    def readValue(self, value_name): 
        try: 
            value, type_ = winreg.QueryValueEx(self.key_handle, value_name)
            return value
        except FileNotFoundError: 
            return None
        

def findRegisteredServerPath(Author, Name): 
    key_name = "Software\\%s\\%s" % (Author, Name)
    with WinRegKey(winreg.HKEY_CURRENT_USER, key_name) as key: 
        exe_path = key.readValue("ExePath")
        if exe_path is None: 
            exe_path = key.readValue("ExeFile")
        win_cls = key.readValue("WinCls")
        return exe_path, win_cls

def findWinByClassName(win_cls): 
    win_list = []
    def _callback(hwnd, user_arg):
        win_list, cls_name = user_arg
        if cls_name == win32gui.GetClassName(hwnd): 
            win_list.append(hwnd)
            #  return False # stop enum  # gives an error ????
        return True # continue
    
    win32gui.EnumWindows(_callback, (win_list, win_cls) ) 
    return 0 if len(win_list)==0 else win_list[0]  

def findWindow(win_cls): 
    return findWinByClassName(win_cls)
    #return win32gui.FindWindow(win_cls, None)  # some problems in old version 
    
def summonServerWin(Author, Name): 
    exe_path, win_cls = findRegisteredServerPath(Author, Name) 
    win = findWindow(win_cls) 
    if win == 0: 
        subprocess.Popen(exe_path, shell=True, stdout=subprocess.PIPE) 

        for i in range(30): 
            win = findWindow(win_cls) 
            if (win != 0): break
            time.sleep(0.3) 
    return win
             
#------------------------------------------------------------------------------

def readSingle(file): 
    b = file.read(4)
    tpl = struct.unpack("<f", b)
    return tpl[0]

def readDouble(file): 
    b = file.read(8)
    tpl = struct.unpack("<d", b)
    return tpl[0]

def readLongInt(file): 
    b = file.read(4)
    return int.from_bytes(b, byteorder='little', signed=True)

#------------------------------------------------------------------------------

class COPYDATASTRUCT_PCHAR(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),
        ('cbData', ctypes.wintypes.DWORD),
        ('lpData', ctypes.c_char_p) #c_wchar_p
        #formally lpData is c_void_p, but we do it this way for convenience
    ]

#------------------------------------------------------------------------------

def __sendCmd(win, cmd, param): 
    win32api.SendMessage(win, WM_SIGLOADER_CMD, cmd, param)

def sendLoaderCmd(win=None, mode=None, device=None, dtype=None, 
                  close_mmap=False, show_server=None, ping=None):
    if win is None: 
        win = summonServerWin('Reonid', 'SignalLoader')

    if   mode is None:  pass
    elif mode == 'file':  __sendCmd(win, CMD_CHANGE_TRANSPORT_MODE, 1)  
    elif mode == 'mmap':  __sendCmd(win, CMD_CHANGE_TRANSPORT_MODE, 0) 
    else: raise SignalLoaderError("Invalid mode: %s" % str(mode))
        
    if   device is not None: 
        device = device.lower()
    
    if   device is None: pass
    elif device == 't10':     __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_T10)   
    elif device == 't-10':    __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_T10)   
    elif device == 'tjii':    __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_TJII)  
    elif device == 'tj-ii':   __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_TJII)  
    elif device == 'tj2':     __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_TJII)  
    elif device == 'tj-2':    __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_TJII)  
    elif device == 'compass': __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_COMPASS)  
    elif device == 'liven':   __sendCmd(win, CMD_CHANGE_DEVICE, DEVICE_LIVEN)  
    else: raise SignalLoaderError("Invalid device: %s" % str(device))

    if   dtype is None:  pass
    elif dtype ==   'float32':  __sendCmd(win, CMD_CHANGE_TYPEID, TYPE_SINGLE)  
    elif dtype == np.float32:   __sendCmd(win, CMD_CHANGE_TYPEID, TYPE_SINGLE)  
    elif dtype ==   'float64':  __sendCmd(win, CMD_CHANGE_TYPEID, TYPE_DOUBLE) 
    elif dtype == np.float64:   __sendCmd(win, CMD_CHANGE_TYPEID, TYPE_DOUBLE) 
    else: raise SignalLoaderError("Unsupported dtype: %s" % str(dtype))
    
    if close_mmap: win32api.SendMessage(win, WM_SIGLOADER_CMD, CMD_CLOSE_MMAP, 0) 
    
    if   show_server is None:  pass
    elif show_server is True:  __sendCmd(win, CMD_SHOW_SERVER, 1)
    elif show_server is False: __sendCmd(win, CMD_SHOW_SERVER, 0)
    else: pass
    
    if ping: 
        ans = win32api.SendMessage(win, WM_MY_SERVER_PING, 133, 204)
        return ans 

def loadSignal(shot, sig, **kwargs): 
    win = summonServerWin('Reonid', 'SignalLoader')
    if win is None: raise SignalError('Cannot find SignalLoader. Please start it manually')
    
    cmd_str = sig + '\0' # just for the case
    cmd_str = cmd_str.encode('ascii')
    
    cds = COPYDATASTRUCT_PCHAR()
    cds.dwData = shot
    cds.cbData = ctypes.sizeof(ctypes.create_string_buffer(cmd_str))
    cds.lpData = ctypes.c_char_p(cmd_str)

    #sendLoaderCmd(win, mode='mmap', device='tjii', dtype='float32')
    sendLoaderCmd(win, **kwargs)
    
    data_length = win32api.SendMessage(win, win32con.WM_COPYDATA, 0, ctypes.addressof(cds))  
    
    #if data_length == 0: return None
    if data_length == 0: raise SignalError('Cannot load signal #%d %s' % (shot, sig) )     
     
    with mmap.mmap(-1, data_length, tagname=SIG_MMAP_NAME) as mm:  
        readLongInt(mm)  #_signature = readLongInt(mm)
        readLongInt(mm)  #_total_size = readLongInt(mm)
        readLongInt(mm)  #_shot = readLongInt(mm)
        mm.read(512)     #_signame_as_bytes = mm.read(512)
        L = readLongInt(mm)
        data_type = readLongInt(mm)
        
        if data_type == TYPE_SINGLE: 
            t0 = readSingle(mm)
            t1 = readSingle(mm)
            data = np.empty((2, L), dtype = np.float32)
            data[0, :] = np.linspace(t0, t1, L)          
        
            #for i in range(L): data[1, i] = readSingle(mm) 
            buffer = mm.read(L*4)
            data[1, :] = np.frombuffer(buffer, np.float32, L)

        elif data_type == TYPE_DOUBLE: 
            t0 = readDouble(mm)
            t1 = readDouble(mm)
            data = np.empty((2, L), dtype = np.float64)
            data[0, :] = np.linspace(t0, t1, L)          
        
            #for i in range(L): data[1, i] = readDouble(mm)             
            buffer = mm.read(L*8)
            data[1, :] = np.frombuffer(buffer, np.float64, L)
        else:
            raise SignalError('Invalid data type #%d %s' % (shot, sig) ) 

     
    sendLoaderCmd(win, close_mmap=True)
    
    return data 

def set_default_kwarg(arg_name, def_val, arg_specification, kwargs_dict): 
    if callable(arg_specification): 
        arg_specification = inspect.getfullargspec(arg_specification)
    
    if arg_name in arg_specification: 
        if not arg_name in kwargs_dict: 
            kwargs_dict[arg_name] = def_val

def set_default_kwargs(func_or_arg_specification, kwargs, new_defaults): 
    if callable(func_or_arg_specification): 
        arg_specification = inspect.getfullargspec(func_or_arg_specification)
    else:    
        arg_specification = func_or_arg_specification

    for key in new_defaults:
        val = new_defaults[key]
        set_default_kwarg(key, val, arg_specification.args, kwargs)
    
    
#------------------------------------------------------------------------------

def __test(): 
    sendLoaderCmd(device='tjii', mode='mmap', dtype='float32')
    sendLoaderCmd(show_server=False)
    
    sig = loadSignal(44381, "HIBPII::Itot{slit3, E=%TJIIEBEAMII%}", dtype='float64')
    plt.plot(sig[0,:], sig[1,:])
        
    
if __name__ == '__main__': 
    __test() 


    