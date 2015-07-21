# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:11:41 2014

@author: Parke
"""

import matplotlib.pyplot as plt
from mypy.specutils import plot as specplot
import io
import database as db
import utils
import reduce
import numpy as np

stars = db.stars

def texname(star):
    with open(db.target_list_tex) as f:
        texnames = f.read().splitlines()

    i = stars.index(star)
    return texnames[i]

def plotrange(spectbl, w0, w1, *args, **kwargs):
    """
    Same as spectstep, but restricted to a given range.
    """
    keep = (spectbl['w1'] > w0) & (spectbl['w0'] < w1)
    specstep(spectbl[keep], *args, **kwargs)

def comparespecs(stars, **kwargs):
    """
    Parameters
    ----------
    binfunc : function or 'all', optional
        Function for regriding spectbls (such as evenbin or powerbin from
        reduce). Default is powerbin from 1000 to 5000 AA with R=500.0
        Set to 'all' to use the entire spectrum.
    axkw
    """
    if 'binfunc' in kwargs:
        binfunc = kwargs['binfunc']
        del kwargs['binfunc']
    else:
        binfunc = lambda s: reduce.powerbin(s, R=5000.0, lo=1100.0, hi=5000.0)

    plts = []
    for star in stars:

        # read in panspectrum
        specfile = db.panpath(star)
        spec = io.read(specfile)[0]

        # interpolate it onto the desired bins
        if binfunc != 'all':
            spec = binfunc(spec)

        # plot
        plts.append(normspec(spec, **kwargs))

    plt.legend(plts, stars)

def earth_from_bol_axis(ax_bol):
    """Given an axis that has bolomoetric-normalized flux as the y-axis,
    add another y-axis on the right side that gives the Earth-equivalent flux.
    """

    # add an axis for equivalent Earth-insolation flux
    ax_earth = ax_bol.twinx()

    # add the appropriate label
    ax_earth.set_ylabel("Earth-Equivalent Flux [erg/s/cm$^2$/$\AA$]")

    # and make it update when the other updates
    def update_earth_ax(ax_bol):
        y0b, y1b = ax_bol.get_ylim()
        y0e, y1e = map(utils.bol2sol, [y0b, y1b])
        ax_earth.set_ylim(y0e, y1e)
        ax_earth.figure.canvas.draw()
    ax_bol.callbacks.connect("ylim_changed", update_earth_ax)

    # now update it for the first time
    update_earth_ax(ax_bol)

    return ax_earth

def normspec(spectbl, **kwargs):
    """Normalize the spectrum by its bolometric flux and plot with the
    appropriate y-axis label."""

    # compute bolometrically normalized fluxes
    spectbl = utils.add_normflux(spectbl)

    # plot up the normalized flux
    nplt = specstep(spectbl, key='normflux', **kwargs)
    eplt = specstep(spectbl, key='normerr', color=nplt.get_color(), ls=':',
                    **kwargs)

    # add the appropriate y label
    norm_eqn = ('$( F_\lambda  / '
                '\int_{0}^{\infty}F_\lambda d\lambda )$')
    plt.ylabel('Normalized Flux ' + norm_eqn +  ' [$\AA^{-1}$]')

    return nplt, eplt

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
    ylabel : str, optional
        Label for the y-axis. Default is '' unless key=='flux'.
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
        spectbls = io.read(spectbl)
        return [specstep(s, *args, **kwargs) for s in spectbls]
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
    if 'ylabel' in kwargs:
        ylbl = kwargs['ylabel']
        del kwargs['ylabel']
    else:
        ylbl = 'Flux [erg/s/cm$^2$/$\AA$]' if key == 'flux' else ''

    #parse data from table
    w0, w1, f, e = np.array([spectbl[s] for s in ['w0','w1', key, 'error']])

    wbins = np.array([w0, w1]).T

    #plot flux
    fplt = specplot(wbins, f, *args, **kwargs)
    plt.xlabel('Wavelength [$\AA$]')
    plt.ylabel(ylbl)

    #plot error
    if err:
#        if 'alpha' not in kwargs: kwargs['alpha'] = 0.4
        kwargs['ls'] = ':'
        if 'color' not in kwargs: kwargs['color'] = fplt.get_color()
        eplt = specplot(wbins, e, *args, **kwargs)
        return fplt, eplt
    else:
        return fplt