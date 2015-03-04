# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:11:41 2014

@author: Parke
"""
from numpy import array
import matplotlib.pyplot as plt
from mypy.specutils import plot as specplot
import io

def plotrange(spectbl, w0, w1, *args, **kwargs):
    """
    Same as spectstep, but restricted to a given range.
    """
    keep = (spectbl['w1'] > w0) & (spectbl['w0'] < w1)
    specstep(spectbl[keep], *args, **kwargs)

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
        Whether to plot errors. Default is false unless key=='flux'.
    *args, **kwargs
        Input to be passed to plot.

    Returns
    -------
    plts : list
        The plot object(s) -- one for each contiguous region of the spectrum.
    errplts : list, optional
        Same as above, but for errors, if plotted.
    """
    if type(spectbl) is str:
        spectbl = io.read(spectbl)
    if 'key' in kwargs:
        key = kwargs['key']
        del kwargs['key']
    else:
        key = 'flux'
    if 'err' in kwargs:
        err = kwargs['err']
        del kwargs['err']
    else:
        err = True if key == 'flux' else False

    #parse data from table
    w0, w1, f, e = array([spectbl[s] for s in ['w0','w1',key,'error']])

    wbins = array([w0, w1]).T

    #plot flux
    fplt = specplot(wbins, f, *args, **kwargs)
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel('Flux [erg/s/cm$^2$/$\AA$]')

    #plot error
    if err:
#        if 'alpha' not in kwargs: kwargs['alpha'] = 0.4
        kwargs['ls'] = ':'
        if 'color' not in kwargs: kwargs['color'] = fplt.get_color()
        eplt = specplot(wbins, e, *args, **kwargs)
        return fplt, eplt
    else:
        return fplt