# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:45:17 2019

@author: reonid

Exports:  
    Signal

    signal(shot, signalName, source='sigloader')
                                    'cache'
                                    'file'
                                    'TS'
    iter_ranges(array, condition)
    find_ranges(array, condition)
    widen_mask, narrow_mask(mask, left, right)
    idx2x(indices, xarray)
    

"""

#import inspect
import numbers # Number, Complex, Real, Rational, Intergral

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import loadsig
import bcache


class Signal: 
    def __init__(self, xy=(None, None), copy=False):
        self.shot = None        
        self.name = ''
        # self.dev = None
        self.t, self.y = xy
        if copy: 
            self.t = deepcopy(self.t)
            self.y = deepcopy(self.y)

    def __iter__(self):  
        '''
        Allows to unpack Signal object as
            x, y = sig
        '''
        yield self.t
        yield self.y

    def __getitem__(self, rng):  
        '''
        Allows to make slice 
            fragment = sig[10:20]
        '''
        if isinstance(rng, slice): 
            xx = self.x[rng]
            yy = self.y[rng]
            fragm = Signal((xx, yy))         
            fragm.shot = self.shot
            fragm.name = self.name
            return fragm
        else: 
            #return (xx, yy)
            if   rng == 0: return self.x
            elif rng == 1: return self.y

    # aliase:  x = t
    def get_x(self): return self.t    
    def set_x(self, x): self.t = x    
    x = property(get_x, set_x)

    def load(self, shot, name, **kwargs): 
        self.t, self.y = loadsig.loadSignal(shot, name, **kwargs)
        self.shot = shot
        self.name = name
        
    def loadfile(self, filename): 
        if filename.lower().endswith('.cache'): 
            self.t, self.y = bcache.loadCacheFile(filename)
            _, self.shot, self.name = bcache.parseCacheFileName(filename)
        else: 
            self.loadtxt(filename)

    def loadtxt(self, filename, cols=(0, 1)): 
        data = np.loadtxt(filename)
        xcol, ycol = cols
        self.t = data[xcol, :]
        self.y = data[ycol, :]
    
    def loadcache(self, shot, name, **kwargs):
        self.t, self.y = bcache.loadCachedSignal(shot, name)
        self.shot = shot
        self.name = name
                
    def plot(self):
        plt.plot(self.x, self.y)
        
    def fragment(self, t0, t1=None): 
        if t1 is None: 
            t0, t1 = t0       # first arg can be tuple (t0, t1)
        j0 = np.searchsorted(self.x, t0)
        j1 = np.searchsorted(self.x, t1)
        return self[j0:j1+1]  # endpoint not included

    def histogram(self, bins=None, **kwargs): 
        if bins is None: 
            L = len(self.y)
            bins = int(2*L**0.35)+1

        cnts, bin_edges = np.histogram(self.y, bins, **kwargs)
        
        result = HistogramSignal(bin_edges, cnts)
        result.name = 'histogram ' + self.name
        result.shot = self.shot
        return result
        
    def zipxy(self): 
        return zip(self.x, self.y)
    
    def resampleas(self, refsig): 
        '''
        ??? Return new signal or modify self ???
        '''
        
        refx = refsig.x if isinstance(refsig, Signal) else refsig
        
        new_x = deepcopy(refx)
        new_y = np.interp(new_x, self.x, self.y)
        self.x, self.y = new_x, new_y
        #result = Signal((new_x, new_y))
        #result.name = self.name
        #result.shot = self.shot
        #return result
        
#------------------------------------------------------------------------------

class HistogramSignal(Signal): 
    def __init__(self, bin_edges, cnts): 
        L = len(cnts)
        xx = np.zeros(L)
        for i in range(L): 
            xx[i] = 0.5*( bin_edges[i] + bin_edges[i+1] )
        
        yy = cnts
        super().__init__((xx, yy))
        
        self._bin_edges = bin_edges

    @property 
    def bins(self):       
        #L = len(self.y)
        #for i in range(L-1): 
        #    yield (self._bin_edges[i], self._bin_edges[i+1])
        for a, b in zip(self._bin_edges, self._bin_edges[1:]): 
            yield (a, b)

#------------------------------------------------------------------------------

def signal(shot, name, source='sigloader', **kwargs): 
    sig = Signal()
    if source == 'sigloader': 
        sig.load(shot, name, **kwargs)
    elif source == 'cache': 
        sig.loadcache(shot, name)
    elif source == 'file': 
        sig.loadtxt(name, **kwargs)
    elif (source == 'TS')or(source == 'thomson'): 
        sig.shot = shot
        sig.name = name
        sig.x, sig.y = bcache.loadThomsonSignal(shot, name)
    return sig


def iter_ranges(seq, condition): 
    '''
    start, fin, cond_value, bound = iter_ranges(sec, condition)
    divides sequence on ranges with the same value of condition 
    boundary = 0 for first range
               1 for last range
               None for the rest
    '''
    inside = False
    fin = 0
    start, last = None, None
    for i, value in enumerate(seq): 
        last = i
        ok = condition(value)
        if (not inside)and(ok): 
            inside = True
            start = i
            if start > fin: 
                yield fin, i, False, (0 if fin == 0 else None)
            fin = None
            
        if (inside)and(not ok): 
            inside = False
            fin = i
            yield start, fin, True, (0 if start == 0 else None)
            start = None

    if start is not None: 
        yield start, last+1, True, 1
    elif fin is not None: 
        yield fin, last+1, False, 1


def find_ranges(seq, condition): 
    return [(i0, i1) for i0, i1, ok, bnd in iter_ranges(seq, condition) if ok] 
        
    
def widen_mask(mask, left, right): 
    '''
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
                     |
                     V
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    '''
    L = len(mask)
    result = deepcopy(mask)
    for i0, i1, ok, bnd in iter_ranges(mask, lambda x: x == 1): 
        if not ok: 
            if bnd != 0:
                result[i0 : min(L, i0 + right)] = True
            if bnd != 1:
                result[max(0, i1 - left) : i1] = True
                
    return result

def narrow_mask(mask, left, right):   
    '''
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
                     |
                     V
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    '''
    L = len(mask)
    result = deepcopy(mask)
    for i0, i1, ok, bnd in iter_ranges(mask, lambda x: x): 
        if ok: 
            if bnd != 0:
                result[i0 : min(L, i0 + left)] = False
            if bnd != 1:
                result[max(0, i1 - right) : i1] = False
                
    return result

def ispair(arg, dtype=None):
    if isinstance(arg, tuple): 
        if len(arg) == 2: 
            if (dtype is None): 
                return True
            else: 
                a, b = arg
                return isinstance(a, dtype)and isinstance(b, dtype)
    return False        

def idx2x(idx, xarray, tuple_as_range=True):  
    '''
    Transform indices to x-values
    Accepts indices, NumPy arrays of indices, 
    tuples of indices of lists, lists of indices or tuples 
        
    idx2x(5, x) = x[5]
    idx2x([1, 2], x) = [x[1], x[2]]
    idx2x((1, 2, 3), x) = (x[1], x[2], x[3])
    
    Tuples of length = 2 treated as open ranges (last element is out of range): 
    idx2x((1, 5), x) = (x[1], x[5-1])
        
    idx2x([(1, 5), (3, 8)], x) = [(x[1], x[5-1]), (x[3], x[8-1])]
    
    idx can be NumPy array of any dimension
    idx = np.array([1, 2, 3])  
    idx = np.array([1, 2], [7, 8])  
    idx = np.array([(1, 8) (3, 10)])
    
    '''
    if isinstance(xarray, Signal): 
        xarray = xarray.x

    if isinstance(idx, numbers.Integral): 
        return xarray[idx]
    elif isinstance(idx, np.ndarray): 
        #return array[idx]
        f = lambda i: idx2x(i, xarray)
        f = np.vectorize(f)
        return f(idx)
    elif ispair(idx, numbers.Integral):
        start, fin = idx
        if tuple_as_range: 
            return (xarray[start], xarray[fin-1])
        else: 
            return (xarray[start], xarray[fin])
    else: 
        generator = (idx2x(i, xarray) for i in idx)
        return idx.__class__(generator)
        

    
if __name__ == '__main__':

    #sig = signal(44381, "HIBPII::Itot{slit3, E=%TJIIEBEAMII%}")
    
    #sig = signal(44381, "Densidad2_", source='cache')
    #sig = signal(29385, "PerfilNe", source='thomson')
    sig = signal(49878, "Te", source='TS')
    sig.plot()


