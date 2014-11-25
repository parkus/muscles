# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:11:41 2014

@author: Parke
"""
from numpy import isclose, array, split, nonzero, logical_not
from my_numpy import lace
import matplotlib.pyplot as plt
import io
from os import path

def specstep(spectbl, *args, **kwargs):
    """
    Plot the spectrum using a stairstep curve and preserving any gaps.
    
    Parameters
    ----------
    spectbl : astropy table
        Spectrum
    err : {True|False}, optional
        Whether to plot errors. Default is false.
    *args, **kwargs
        Input to be passed to plot.
        
    Returns
    -------
    plt : object
        The plot object(s) -- one for each contiguous region of the spectrum.
    vlns : object, optional
        The corresponding vlines object(s) if err == True.
    """
    if 'err' in kwargs:
        err = kwargs['err']
        del kwargs['err']
    else:
        err = False
    
    #make into an array for more concise manipulation
    s = array([spectbl[s] for s in ['w0','w1','flux','error']])
    
    #split the spectrum at any gaps
    isgap = logical_not(isclose(s[0,1:], s[1,:-1]))
    gaps = nonzero(isgap)[0] + 1
    slist = split(s, gaps, 1)
    
    plts,vlns = [],[]
    for s in slist:
        w0,w1,f,e = s
        
        #make vectors that will plot as a stairstep
        w = lace(w0, w1)
        f = lace(f, f)
        plts.append(plt.plot(w, f, *args, **kwargs))
        
        if err:
            e = lace(e, e)
            plts.append(plt.plot(w, e, *args, **kwargs))
    
    return plts
    
def cyclespec(files):
    plt.ioff()
    for f in files:
        specs = io.read(f)
        for spec in specs:
            specstep(spec)
        plt.title(path.basename(f))
        plt.xlabel('Wavelength [$\AA$]')
        plt.ylabel('Flux [erg/s/cm$^2$/$\AA$]')
        plt.show()
    plt.ion()