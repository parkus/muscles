# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:11:41 2014

@author: Parke
"""
from numpy import array
import matplotlib.pyplot as plt
from mypy.specutils import plot as specplot

def specstep(spectbl, *args, **kwargs):
    """
    Plot the spectrum using a stairstep curve and preserving any gaps.
    
    Parameters
    ----------
    spectbl : astropy table
        Spectrum
    key : str, optional
        Which table column to plot. Defaults to flux.
    err : {True|False}, optional
        Whether to plot errors. Default is false.
    *args, **kwargs
        Input to be passed to plot.
        
    Returns
    -------
    plts : list
        The plot object(s) -- one for each contiguous region of the spectrum.
    errplts : list, optional
        Same as above, but for errors, if plotted. 
    """
    if 'err' in kwargs:
        err = kwargs['err']
        del kwargs['err']
    else:
        err = True
    if 'key' in kwargs:
        key = kwargs['key']
        del kwargs['key']
    else:
        key = 'flux'
    
    #parse data from table
    w0, w1, f, e = array([spectbl[s] for s in ['w0','w1',key,'error']])
    
    wbins = array([w0, w1]).T
    
    #plot flux
    plts = specplot(wbins, f, *args, **kwargs)
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Flux [erg/s/cm$^2$/$\AA$]')
    
    #plot error
    if err:
        if 'alpha' not in kwargs: kwargs['alpha'] = 0.3
        errplts = specplot(wbins, e, *args, **kwargs)
        return plts, errplts
    else:
        return plts