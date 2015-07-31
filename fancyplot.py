import mypy.plotutils as pu
import plot as mplt
import matplotlib.pyplot as plt
import scicatalog.scicatalog as sc
import database as db
import numpy as np
import os
import mypy.my_numpy as mnp
from reduce import auto_curve
import settings
import io, reduce
import matplotlib as mpl
import itertools
import uneven.functions as un
import spectralPhoton.functions as sp
import numpy as np

def stars3DMovieFrames(size, azRate=1.0, altRate=0.0, frames=360, dirpath='muscles_stars_movie_frames'):
    from mayavi import mlab

    starprops = sc.SciCatalog(db.proppath, readOnly=True)
    ra, dec, dist, r, T, names = [starprops.values[s] for s in ['RA', 'dec', 'dist', 'R', 'Teff', 'name txt']]

    labels = names.values

    az = 0.0
    alt = 60.0
    rcam = 72.0
    focalPt = np.zeros(3)
    view = [az, alt, rcam, focalPt]
    fig = pu.stars3d(ra, dec, dist, T, r, size=size, labels=labels, view=view)

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    for i in xrange(frames):
        filename = '{:03d}.png'.format(i)
        filepath = os.path.join(dirpath, filename)
        mlab.savefig(filepath, figure=fig, size=size)
        az += azRate
        alt += altRate
        mlab.view(az, alt, rcam, focalPt, figure=fig)


def lightcurveCompendium(stars='hosts', figure=None, flarecut=2.0, flarelabel='SiIV', dt=30.0, colorful=False):
    """
    Create a compendium of lightcurves for the specified stars highlighting flares.
    """
    fig = plt.gcf() if figure is None else figure
    if colorful:
        colors = itertools.cycle(['b', 'g', 'r'])
    else:
        colors = itertools.cycle(['k'])
    inst = 'hst_cos_g130m'
    if stars == 'hosts':
        stars = filter(lambda s: len(db.findfiles('u', inst, 'corrtag_a', s)) >= 4, db.observed)
    starprops = sc.SciCatalog(db.proppath, readOnly=True)

    ## SET UP AXES
    ## -----------
    # setup axes with just bottom axis showing
    fig.set_facecolor('w')
    fig.add_axes((0.23, 0.14, 0.72, 0.84), frameon=False)
    ax = fig.axes[0]
    xax = ax.get_xaxis()
    xax.tick_bottom()
    yax = ax.get_yaxis()
    yax.set_visible(False)
    ax.set_clip_on(False)

    fntsz = mpl.rcParams['font.size']
    spacedata = (fntsz / 72.0 / fig.get_figwidth()) / 0.8 * 14000

    # common keywords to use in errorbar plot
    ekwds = dict(fmt='.', capsize=0.0)
    alphaNF = 0.1 if colorful else 0.4

    ## MAKE LIGHTCURVES
    ## ----------------
    bands = settings.flare_bands[inst]
    offset, offsets = 0.0, []
    for star, color in zip(stars, colors):
        # get flare info
        flares, bands = io.readFlareTbl(star, inst, flarelabel)

        curve = auto_curve(star, inst, dt=30.0, bands=bands, groups=[range(len(bands))])
        t0, t1, cps, err = zip(*curve)[0]

        t = (t0 + t1) / 2.0

        # get rid of gaps
        s, j, _ = mnp.shorten_jumps(t, maxjump=10*dt, newjump=30*dt)

        # identify flare pts
        flares = flares[flares['PEWratio'] > flarecut]
        flare_ranges = np.array([flares['start'], flares['stop']]).T
        flarepts = mnp.inranges(t, flare_ranges)

        # normalize the data to median
        med = np.median(cps[~flarepts])
        y = cps/med
        y -= 1.0
        e = err/med

        # normalize data to max - min
        ymax = np.max(y)
        y /= ymax
        e /= ymax

        # cull negative outliers
        good = y > -ymax
        s, y, e, flarepts = [a[good] for a in [s, y, e, flarepts]]

        # offset data in y
        offset = offset - np.min(y - e)
        offsets.append(offset)
        yo = y.copy() + offset

        # plot data
        ax.errorbar(s[~flarepts], yo[~flarepts], e[~flarepts], alpha=alphaNF, color=color, **ekwds)
        ax.axhline(offset, linestyle='--', color='k', alpha=0.5)
        flarecolor = 'r' if color == 'k' else color
        ax.errorbar(s[flarepts], yo[flarepts], e[flarepts], color=flarecolor, **ekwds)

        # make arrow
        xy = (0.0, offset)
        xytext = (0.0, offset + 1.1)
        arrowprops = dict(arrowstyle='<-', color=color)
        ax.annotate('', xy=xy, xytext=xytext, arrowprops=arrowprops)

        # label arrow
        ymaxstr = '{:4.1f}'.format(ymax + 1.0)
        ax.text(-0.5*spacedata, offset, '1.0', va='bottom', ha='right', color=color, fontsize=fntsz*0.8)
        ax.text(-0.5*spacedata, offset + 1.1, ymaxstr, va='top', ha='right', color=color, fontsize=fntsz*0.8)

        # label star
        starlbl = starprops['name tex'][star]
        ax.text(-3.0*spacedata, offset + 0.5, starlbl, va='center', ha='right', color=color)

        # increase offset
        offset += np.max(y + e)
        offset += 0.3

    ax.autoscale(axis='both', tight=True)

    # add x axis line and label
    ax.axhline(0, color='k')
    ax.set_xlabel('Time, Observation Gaps Shortened (s)')

def cumFlareFreq(band='SiIV', inst='cos_g130m', stars='all', metric='both', ax=None, fits=False):
    flarecat = reduce.combine_flarecats(band, inst, stars=stars)
    ax = plt.gca() if ax is None else ax
    ax.set_ylabel('Cumulative Frequency, $\\nu$ [d$^{-1}$]')

    xlabelPEW = 'Luminosity-Normed Flare Energy, $P$ (Photometric Equiv. Width) [s]'
    xlabelE = 'Absolute Flare Energy, $E$ [erg]'
    if metric == 'both':
        pP = ax.loglog(flarecat['PEW'], flarecat['cumfreqPEW'], 'bo')
        ax.set_xlabel(xlabelPEW, color='b')
        for tl in ax.get_xticklabels():
            tl.set_color('b')
        if fits:
            x, y, lbl = _plotPowFit(flarecat['PEW'], flarecat['cumfreqPEW'], ax, '$P$', color='b')
            ax.text(x*1.3, y*1.3, lbl, color='b')

        ax2 = ax.twiny()
        pE = ax2.loglog(flarecat['energy'], flarecat['cumfreqE'], 'rd')
        ax2.set_xlabel(xlabelE, color='r')
        ax2.xaxis.set_label_position('top')
        for tl in ax2.get_xticklabels():
            tl.set_color('r')
        if fits:
            x, y, lbl = _plotPowFit(flarecat['energy'], flarecat['cumfreqE'], ax, '$E$', color='r')
            ax.text(x/1.3, y/1.3, lbl, color='r', va='top', ha='right')

        return ax2

    if metric == 'PEW':
        ax.loglog(flarecat['PEW'], flarecat['cumfreqPEW'], 'ko')
        ax.set_xlabel(xlabelPEW)
        if fits:
            x, y, lbl = _plotPowFit(flarecat['PEW'], flarecat['cumfreqPEW'], ax, '$P$', color='k')
            ax.text(1.3*x, 1.3*y, lbl)

    if metric == 'energy':
        ax.loglog(flarecat['energy'], flarecat['cumfreqE'], 'ko')
        ax.set_xlabel(xlabelE)
        if fits:
            x, y, lbl = _plotPowFit(flarecat['energy'], flarecat['cumfreqE'], ax, '$E$', color='k')
            ax.text(1.3*x, 1.3*y, lbl)

def _plotPowFit(x, freq, ax, yvar, **kwargs):
    a, err = un.powfit(x)
    # clooge up a normalization factor by minimizing the squared deviations
    norm = np.sum(freq * x**(-a)) / np.sum(x**(-2*a))
    xplt = np.array([x.min(), x.max()])
    fplt = norm * xplt**-a
    ax.plot(xplt, fplt, **kwargs)
    xmid, fmid = np.sqrt(xplt.prod()), np.sqrt(fplt.prod())
    label = '$\\nu \sim %s^{-%.2f \\pm %.2f}$' % (yvar, a, err)
    return xmid, fmid, label

def spectrumMovieFrames(star, inst, band, trange, dt, smoothfac, axspec, axcurve, folder, dpi=80, velocityplot=False,
                        reftrange=None, dryRun=False, ylim=None):
    ph, photons = io.readphotons(star, inst)
    band, trange, reftrange = map(np.asarray, [band, trange, reftrange])

    fig = axcurve.get_figure()
    figwidth = fig.get_figwidth()
    axwidth = axcurve.get_position().width * figwidth
    axPix = axwidth * dpi

    def goodN(Nphotons):
        n = 100 # SN of 10
        if Nphotons / n > axPix: # don't need to sample more finely than the pixel scale
            n = Nphotons / axPix
        elif Nphotons / n < 20: # want at least some resolution
            n = max(Nphotons/20, 9) # but no less than SN of 3ish
        return n

    # re-reference times to start of time range
    tref = trange[0]
    photons['time'] -= tref
    if reftrange is not None: reftrange -= tref
    trange -= tref

    if velocityplot:
        velocify = lambda w:  (w - velocityplot)/velocityplot * 3e5
        vband = velocify(band)

    p = photons
    tkeep = [max(p['time'][0], -trange[1]*0.5),  min(p['time'][-1], trange[1]*1.5)]
    dw = band[1] - band[0]
    wkeep = [band[0] - 5*dw, band[1] + 5*dw]

    # get rid of superfluous counts
    keep = mnp.inranges(p['time'], tkeep) & mnp.inranges(p['wavelength'], wkeep)
    p = p[keep]

    ## make lightcurve and set up initial plot
    nlc = goodN(np.sum(mnp.inranges(p['wavelength'], band) & mnp.inranges(p['time'], trange)))
    t0, t1, lc, lcerr = sp.smooth_curve(p['time'], p['wavelength'], p['epera'], nlc, bands=[band], trange=tkeep)
    tlc = (t0 + t1) / 2.0
    axcurve.set_xlim(trange)
    axcurve.set_xlabel('Time [s]')
    axcurve.set_ylabel('Integrated Flux \n[erg cm$^{-2}$ s$^{-1}$]')
    inrange = mnp.inranges(tlc, trange)
    pu.errorpoly(tlc[inrange], lc[inrange], lcerr[inrange], 'k-', ax=axcurve, alpha=0.3, ealpha=0.15)

    # make spectrum frames
    T = dt*smoothfac
    nframes = int(round((trange[1] - trange[0] - T) / dt))
    t1s = np.linspace(trange[0] + T, trange[1], nframes)
    t0s = t1s - T
    wList, specList, errList = [], [], []
    for t0, t1 in zip(t0s, t1s):
        inInterval = mnp.inranges(p['time'], [t0, t1])
        pt = p[inInterval]
        n = goodN(np.sum(mnp.inranges(pt['wavelength'], band)))
        w0, w1, spec, err = sp.smooth_spec(pt['wavelength'], pt['epera'], n, wkeep)
        wList.append((w0 + w1)/2.0)
        specList.append(spec / T)
        errList.append(err / T)

    ## set up spectrum plot
    if velocityplot:
        axspec.set_xlabel('Doppler Velocity [km s$^{-1}$]')
        axspec.set_xlim(vband)
    else:
        axspec.set_xlabel('Wavelength [$\AA$]')
        axspec.set_xlim(band)
    axspec.set_ylabel('Flux [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]')
    if ylim is None:
        ymin = min([np.min(s-e) for s,e in zip(specList, errList)])
        ymax = max([np.max(s+e) for s,e in zip(specList, errList)])
        axspec.set_ylim(ymin, ymax)
    else:
        axspec.set_ylim(ylim)

    # compute and plot reference spectrum, if desired
    if reftrange is not None:
        gtis = np.array([ph['gti'].data['start'], ph['gti'].data['stop']]).T - tref
        gt = mnp.range_intersect([reftrange], gtis)
        Tref = np.sum(gt[:,1] - gt[:,0])
        keep = mnp.inranges(photons['time'], reftrange) & mnp.inranges(photons['wavelength'], wkeep)
        pt = photons[keep]
        nref = goodN(np.sum(mnp.inranges(pt['wavelength'], band)))
        w0, w1, spec, _= sp.smooth_spec(pt['wavelength'], pt['epera'], nref, wkeep)
        w = (w0 + w1)/2.0
        if velocityplot:
            w = velocify(w)
        spec = spec / Tref
        axspec.plot(w, spec, 'k-', alpha=0.3)

    # make folder to save frames in if it doesn't exist
    if not os.path.exists(folder):
        os.mkdir(folder)

    ## loop through frames
    if dryRun:
        nframes = dryRun
    for i in range(nframes):
        # plot time range on lightcurve
        span = axcurve.axvspan(t0s[i], t1s[i], color='k', alpha=0.2)
        inrange = mnp.inranges(tlc, [t0s[i], t1s[i]])
        linelc, polylc = pu.errorpoly(tlc[inrange], lc[inrange], lcerr[inrange], 'k-', ax=axcurve, ealpha=0.3)

        # plot spectrum
        w, spec, err = wList[i], specList[i], errList[i]
        inrange = mnp.inranges(w, band)
        inrange = np.nonzero(inrange)[0]
        inrange = np.insert(inrange, [0, len(inrange)], [inrange[0]-1, inrange[-1]+1])
        ww, ss, ee = w[inrange], spec[inrange], err[inrange]
        ss[[0, -1]] = np.interp(band, ww, spec[inrange])
        ee[[0, -1]] = np.interp(band, ww, err[inrange])
        ww[[0,-1]] = band
        if velocityplot:
            ww = velocify(ww)
        linespec, polyspec = pu.errorpoly(ww, ss, ee, 'k-', ax=axspec, ealpha=0.2)

        # save frame
        path = os.path.join(folder, '{:04d}.png'.format(i))
        fig.savefig(path, dpi=dpi)

        # remove plots
        [obj.remove() for obj in [span, linelc, polylc, linespec, polyspec]]


def showFlareStats(star, inst, label, trange, dt=10.0, ax=None):
    if ax is None: ax = plt.gca()

    flareTbl, bands = io.readFlareTbl(star, inst, label)

    # make lightcurve
    groups = [range(len(bands))]
    t0, t1, flux, err = reduce.auto_curve(star, inst, bands, dt, appx=False, groups=groups, fluxed=True)
    t = (t0 + t1) / 2.0

    # narrow lightcurve down to range of interest
    keep = (t1 > trange[0]) & (t0 < trange[1])
    t, flux, err = t[keep], flux[keep], err[keep]

    # keep only largest flare in time range of interest
    keep = (flareTbl['start'] <  trange[1]) & (flareTbl['stop'] < trange[0])
    flareTbl = flareTbl[keep]
    iflare = np.argmax(flareTbl['PEW'])
    flare = flareTbl[iflare]

    # plot lightcurve highlighting flare points
    flarepts = mnp.inranges(t, [flare['start'], flare['stop']])
    ax.errorbar(t[~flarepts], flux[~flarepts], err[~flarepts], 'k.', capsize=0)
    ax.errorbar(t[flarepts], flux[flarepts], err[flarepts], 'r.', capsize=0)

    # reverse-engineer mean flux
    luminosity = flare['energy'] / flare['PEW']
    dist = sc.quickval(db.proppath, star, 'dist')
    mnflux = luminosity / 4 / np.pi / (dist * 3.08567758e18)**2

    # plot mean flux


