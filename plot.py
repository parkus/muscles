# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:11:41 2014

@author: Parke
"""
from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
from mypy.specutils import plot as specplot
from . import rc, io, utils, db
import numpy as np

stars = rc.stars


def texname(star):
    with open(rc.target_list_tex) as f:
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
        binfunc = lambda s: utils.powerbin(s, R=5000.0, lo=1100.0, hi=5000.0)

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
        y0e, y1e = list(map(utils.bol2sol, [y0b, y1b]))
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


def phxCompare(star, wlim=None, maxpow=None, mindw=None, ax=None):
    if ax is None: ax = plt.gca()

    xf = db.findfiles('ir', 'phx', star, fullpaths=True)
    pf = db.panpath(star)

    phx = io.read(xf)[0]
    pan = io.read(pf)[0]
    normfac = pan['normfac'][-1]
    if wlim is not None:
        phx, pan = utils.keepranges(phx, wlim), utils.keepranges(pan, wlim)

    if maxpow or mindw:
        pan = utils.fancyBin(pan, maxpow=maxpow, mindw=mindw)
        phx = utils.fancyBin(phx, maxpow=maxpow, mindw=mindw)

    Fbol = utils.bolo_integral(star)
    pan['normflux'] = pan['flux']/Fbol
    pan['normerr'] = pan['error']/Fbol
    phx['normflux'] = phx['flux']*normfac/Fbol

    line = specstep(pan, key='normflux', label='Panspec', ax=ax)
    specstep(pan, key='normerr', label='Panspec Error', ax=ax, color=line.get_color(), ls=':')
    specstep(phx, key='normflux', label='Phoenix', ax=ax, color='r', alpha=0.5)
    ax.set_xlim(wlim)
    ax.legend(loc='best')


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

    ax = kwargs['ax'] if 'ax' in kwargs else plt.gca()
    key = kwargs.pop('key', 'flux')
    err = kwargs.pop('err', (True if key == 'flux' else False))
    ylbl = kwargs.pop('ylabel', ('Flux [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]' if key == 'flux' else ''))
    autolabel = kwargs.pop('autolabel', True)

    #parse data from table
    w0, w1, f = np.array([spectbl[s] for s in ['w0','w1', key]])

    wbins = np.array([w0, w1]).T

    #plot flux
    if autolabel:
        ax.set_xlabel('Wavelength [$\AA$]')
        ax.set_ylabel(ylbl)

    #plot error
    if err != False:
        e = spectbl['error'] if key == 'flux' else spectbl[key + '_err']

    if err in [True, 'line']:
        fplt = specplot(wbins, f, *args, **kwargs)
        if 'color' not in kwargs: kwargs['color'] = fplt.get_color()
#        if 'alpha' not in kwargs: kwargs['alpha'] = 0.4
        kwargs['ls'] = ':'
        eplt = specplot(wbins, e, *args, **kwargs)
        return fplt, eplt
    elif err == 'poly':
        specplot(wbins, f, err=e, *args, **kwargs)
    else:
        fplt = specplot(wbins, f, *args, **kwargs)
        return fplt


def getcurves():
    """
    Use this just to save time by not recreating curves each time you
    reload the function.
    """
    curves = []
    for star in stars:
        tagfiles = db.findfiles('u', 'corrtag_a', 'cos_g130m', star, fullpaths=True)
        x1dfiles = db.findfiles('u', 'x1d', 'cos_g130m', star, fullpaths=True)

        curve = cv.autocurve(tagfiles, x1dfiles, dt=dt, bands=bands, groups=groups)
        curves.append(curve[0])

    s, e, y, maxnorm, offsets, flarepts = [], [], [], [], [], []
    offset = 0.0
    for curve, flarerng in zip(curves, flarerngs):
        t = (curve['t0'] + curve['t1']) / 2.0
        t = t.data[1:]

        # get rid of gaps
        ss, j, _ = mnp.shorten_jumps(t, maxjump=10*dt)
        s.append(ss)

        # normalize the data to median
        f = curve['cps'].data[1:]
        fmed = np.median(f)
        yy = f/fmed
        yy -= 1.0
        ee = curve['cps err'].data[1:]/fmed

        # normalize data to max - min
        ymax = np.max(yy)
        yy /= ymax
        ee /= ymax

        # identify flare pts
        i = np.arange(len(yy))
        ff = mnp.inranges(i, flarerng)

        # cull negative outliers
        good = yy/ee > -2.0
        ss, yy, ee, ff = [a[good] for a in [ss, yy, ee, ff]]

        y.append(yy)
        e.append(ee)
        flarepts.append(ff)
        maxnorm.append(ymax)

        offset = offset - np.min(yy - ee)
        offsets.append(offset)
        offset += np.max(yy + ee)
        offset += 0.2

    stuff = list(zip(s, y, e, flarepts, maxnorm, offsets))
    return stuff

def setupAxes():
    plt.figure(figsize=[10.666, 5])
    ax = plt.axes(frameon=False)
    xax = ax.get_xaxis()
    xax.tick_bottom()
    yax = ax.get_yaxis()
#    ax.set_xticks([0, 5000, 10000, 15000])
    plt.subplots_adjust(0.02, 0.10, 0.95, 0.95)

    yax.set_visible(False)

    ymin, _ = yax.get_view_interval()

    plt.hlines(0, 0, maxt, lw=2, color='k')

    plt.xlim(-23*label_offset, 14100)
#    plt.ylim(-10.0, None)

    plt.text(maxt/2.0, -0.9, 'Time, Observation Gaps Shortened (s)',
             ha='center', va='center')

def plotLC(stuff, colors, fname):
    setupAxes()

    for (ss, yy, ee, flare, mn, offset), color, star in zip(stuff, colors, stars):
        yo = yy.copy() + offset



        ekwds = dict(fmt='.', capsize=0.0, color=color, ms=3)
        plt.errorbar(ss[~flare], yo[~flare], ee[~flare], alpha=0.2, **ekwds)
        plt.plot([-rng_offset, maxt], [offset]*2, '--', color=color)
        plt.text(-label_offset, offset + 0.5, ml.plot.texname(star),
                 va='center', ha='right', color=color)

        # make arrow
        xy = (-rng_offset, offset)
        xytext = (-rng_offset, offset + 1.1)
        arrowprops = dict(arrowstyle='<-', color=color)
        plt.annotate('', xy=xy, xytext=xytext, arrowprops=arrowprops)

        # label arrow
        mnstr = str(round(mn, 1))
        plt.text(-rng_offset*2.0, offset, '1.0', va='bottom', ha='right',
                 color=color, fontsize=fntsz*0.8)
        plt.text(-rng_offset*2.0, offset + 1.1, mnstr, va='top', ha='right',
                 color=color, fontsize=fntsz*0.8)

        if ekwds['color'] == 'k':
            ekwds['color'] = 'r'
        plt.errorbar(ss[flare], yo[flare], ee[flare], **ekwds)

    plt.xlim(-label_offset - 2000, maxt)
    ax = plt.gca()
    xt = np.array(ax.get_xticks())
    ax.set_xticks(xt[xt >= 0])
    plt.autoscale(axis='y', tight=True)
    plt.savefig(rc.root + '/scratchwork/' + fname, dpi=300)

colors = ['b', 'g', 'r', 'b', 'g', 'r']
blacks = ['k']*6


