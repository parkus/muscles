import mypy.plotutils as pu
import matplotlib.pyplot as plt
import os
import mypy.my_numpy as mnp
import rc, io, reduce, db
import matplotlib as mpl
import itertools
import uneven.functions as un
import spectralPhoton.functions as sp
import numpy as np
import spectralPhoton.functions as sp
from astropy.io import fits

linestyles = itertools.cycle(['-', '--', ':', '.-'])
linefluxlabel = '$\lambda$-Integrated Flux \n[erg cm$^{-2}$ s$^{-1}$]'
fluxlabel = 'Flux [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'

starprops = rc.starprops

def instrumentRanges(star, figure):
    pass



def stars3DMovieFrames(size, azRate=1.0, altRate=0.0, frames=360, dirpath='muscles_stars_movie_frames'):
    from mayavi import mlab

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
        stars = filter(lambda s: len(db.findfiles('u', inst, 'corrtag_a', s)) >= 4, rc.observed)

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
    bands = rc.flare_bands[inst]
    offset, offsets = 0.0, []
    for star, color in zip(stars, colors):
        # get flare info
        flares, bands = io.readFlareTbl(star, inst, flarelabel)

        curve = reduce.auto_curve(star, inst, dt=30.0, bands=bands, groups=[range(len(bands))])
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

def cumFlareFreq(band='SiIV', inst='cos_g130m', stars='all', metric='energy', ax=None, fits=False, flarecut=1.0,
                 **kwargs):
    flarecat = reduce.combine_flarecats(band, inst, flarecut, stars=stars)
    ax = plt.gca() if ax is None else ax
    ax.set_ylabel('Cumulative Frequency, $\\nu$ [d$^{-1}$]')
    ax.set_xscale('log')
    ax.set_yscale('log')

    if metric == 'both':
        cumFlareFreq(band, inst, stars, 'PEW', ax, fits, **kwargs)
        ax2 = ax.twiny()
        ax2.xaxis.set_ticks_position('top')
        ax2.xaxis.set_label_position('top')
        cumFlareFreq(band, inst, stars, 'energy', ax2, fits, ls='--', **kwargs)
        ax2.text(0.95, 0.95, 'dashed - abs. energy\nsolid - phot. equiv. width', ha='right',
                 va='top', transform=ax2.transAxes)
        return
    if metric == 'PEW':
        xkey, var = 'cumfreqPEW', 'P'
        ax.set_xlabel('Luminosity-Normed Flare Energy, $P$ (Photometric Equiv. Width) [s]')
    if metric == 'energy':
        xkey, var = 'cumfreqE', 'E'
        ax.set_xlabel('Absolute Flare Energy, $E$ [erg]')

    flarecat.sort(metric)
    line = ax.step(flarecat[metric], flarecat[xkey], where='post', **kwargs)[0]
    if fits:
        x, y, lbl, fitline = _plotPowFit(flarecat[metric], flarecat[xkey], ax, var, alpha=0.5,
                                         **kwargs)
        fitlbl = ax.text(1.3*x, 1.3*y, lbl)
        return line, fitline, fitlbl
    else:
        return line


def _plotPowFit(x, freq, ax, yvar, **kwargs):
    a, _ = un.powfit(x)

    # clooge up a normalization factor by minimizing the squared deviations
    # norm = np.sum(freq * x**(-a)) / np.sum(x**(-2*a))

    # clooge up a normaliation factor by equating the integrals
    xlo, xhi = x.min(), x.max()
    S = np.sum(np.diff(x) * freq[1:])
    A = -a + 1
    norm = S*A / (xhi**A - xlo**A)

    xplt = np.array([xlo, xhi])
    fplt = norm * xplt**-a
    line = ax.plot(xplt, fplt, **kwargs)[0]
    xmid, fmid = np.sqrt(xplt.prod()), np.sqrt(fplt.prod())
    label = '$\\nu \sim %s^{-%.2f}$' % (yvar, a)

    return xmid, fmid, label, line


def flareCompare(inst='cos_g130m', band='SiIV', nflares=3, ax=None):
    if ax is None: ax = plt.gca()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Normalized Flux')

    flares = reduce.combine_flarecats(band, inst)
    flares = flares[:nflares]
    bands, dt = flares.meta['BANDS'], flares.meta['DT']
    groups = [range(len(bands))]

    mxpeak = max(flares['pkratio'])

    # plot all curves
    lines, labels = [], []
    for flare in flares:
        star = flare['star']
        label = starprops['name tex'][star]

        # curve = reduce.auto_curve(star, inst, bands=bands, dt=dt, appx=False, groups=groups)
        # t0, t1, cps, cpserr = zip(*curve)[0]
        ph, p = io.readphotons(star, inst)
        t0, t1, cps, cpserr = sp.smooth_curve(p['time'], p['wavelength'], p['epsilon'], bands=bands, n=100)

        tmid = (t0 + t1) / 2.0
        t = tmid - flare['peak rel']

        qcps = fits.getval(rc.flarepath(star, inst, 'SiIV'), 'QSCTCPS', ext=1)
        rate, err = cps/qcps, cpserr/qcps

        fac = 10.0**np.floor(np.log10(mxpeak / flare['pkratio']))
        if fac > 1.0:
            label += ' $\\times$%i' % fac
            rate = (rate - 1.0)*fac + 1.0
            err = err*fac

        # line = ax.plot(t, rate, '-')[0]
        # ax.errorbar(t, rate, err, fmt='.', color=line.get_color(), capsize=0)
        line, poly = pu.errorpoly(t, rate, err, ax=ax)
        lines.append(line); labels.append(label)

    ax.legend(lines, labels, loc='best', fontsize='small')

    starts, stops = flares['start rel'] - flares['peak rel'], flares['stop rel'] - flares['peak rel']
    ax.set_xlim(min(starts)*1.5, max(stops))


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
    axspec.set_ylabel(fluxlabel)
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


def showFlareStats(star, inst, label, trange, dt=30.0, ax=None):
    if ax is None: ax = plt.gca()
    trange = np.array(trange)

    flareTbl, bands = io.readFlareTbl(star, inst, label)

    # make lightcurve
    groups = [range(len(bands))]
    curve = reduce.auto_curve(star, inst, bands, dt, appx=False, groups=groups, fluxed=True)
    t0, t1, flux, err = zip(*curve)[0]
    t = (t0 + t1) / 2.0

    # narrow lightcurve down to range of interest
    keep = (t1 > trange[0]) & (t0 < trange[1])
    t, flux, err = t[keep], flux[keep], err[keep]
    ymax = flux.max()

    # keep only largest flare in time range of interest
    keep = (flareTbl['start'] < trange[1]) & (flareTbl['stop'] > trange[0])
    flareTbl = flareTbl[keep]
    iflare = np.argmax(flareTbl['PEW'])
    flare = flareTbl[iflare]

    # reference time to start of flare
    tref = flare['start']
    t, flare['start'], flare['stop'], trange = t - tref, flare['start'] - tref, flare['stop'] - tref, trange - tref

    # plot lightcurve highlighting flare points
    flarepts = mnp.inranges(t, [flare['start'], flare['stop']])
    ax.errorbar(t[~flarepts], flux[~flarepts], err[~flarepts], fmt='ko', capsize=0)
    ax.errorbar(t[flarepts], flux[flarepts], err[flarepts], fmt='rd', capsize=0)

    # reverse-engineer mean flux
    luminosity = flare['energy'] / flare['PEW']
    dist = sc.quickval(rc.proppath, star, 'dist')
    mnflux = luminosity / 4 / np.pi / (dist * 3.08567758e18)**2

    # plot mean flux
    ax.axhline(mnflux, color='gray', ls='--')
    tmid = np.mean(t[~flarepts])
    mnfluxstr = '$\overline{F}$ = %.1e' % mnflux
    ax.text(tmid, mnflux + ymax/20.0, mnfluxstr, ha='center')

    # # plot duration
    # dt = flare['stop'] - flare['start']
    # y = 1.05 * np.max(flux + err)
    # ax.annotate('', xy=(flare['start'], y), xytext=(flare['stop'], y), arrowprops=dict(arrowstyle='<->', color='r'))
    # # ax.arrow(flare['start'], y, dt, 0.0, length_includes_head=True)
    # # ax.arrow(flare['start'], y, dt, 0.0, length_includes_head=True, head_starts_at_zero=True)
    # ax.text(tmidFlare, y*1.02, '{:.0f} s'.format(dt), ha='center', color='r')

    # fill area under flare
    # tmidFlare = np.sum(t[flarepts]*flux[flarepts])/np.sum(flux[flarepts])
    lo = [mnflux] * np.sum(flarepts)
    ax.fill_between(t[flarepts], lo, flux[flarepts], color='r', alpha=0.3)
    y = 0.2 * np.max(flux)
    intlbl = 'equiv. width = {:.0f} s\nenergy = {:.1g} erg'.format(flare['PEW'], flare['energy'])
    ax.text(0.05, 0.95, intlbl, ha='left', va='top', color='r', transform=ax.transAxes)

    # axes
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(linefluxlabel)


def multiBandCurves(star='gj832', inst='cos_g130m', trange=[24400, 24900], n=100, tref=24625.0,
                    bandnames=['NV', 'SiIV', 'SiIII', 'CII', 'cont1380'], ax=None, norms=None, fluxed=False,
                    maxpts=500):
    if ax is None: ax = plt.gca()
    ph, p = io.readphotons(star, inst)
    p['time'] -= tref
    trange = np.array(trange) - tref

    ax.set_xlabel('Time [s]')
    ax.set_xlim(trange)
    if fluxed:
        wtkey = 'epera'
        ax.set_ylabel(linefluxlabel)
    else:
        wtkey = 'epsilon'
        ax.set_ylabel('Count Rate [s$^{-1}$]')
    if norms is not None:
        ax.set_ylabel('Normalized Flux')

    lines, labels = [], []
    for i, name in enumerate(bandnames):
        bands = rc.stdbands.loc[name, 'bands']
        label = rc.stdbands.loc[name, 'name txt']
        Tform = rc.stdbands.loc[name, 'Tform']
        # wavelbl = rc.stdbands.loc[name, 'wave lbl']
        # label += ' ' + wavelbl
        if not np.isnan(Tform):
            label += '  ({:.1f})'.format(Tform)

        t0, t1, rate, err = sp.smooth_curve(p['time'], p['wavelength'], p[wtkey], n=n, bands=bands)
        t = (t0 + t1) / 2.0
        keep = (t1 > trange[0]) & (t0 < trange[1])
        t, rate, err = t[keep], rate[keep], err[keep]

        if len(t) > maxpts:
            keep = un.downsample_even(t, maxpts)
            t, rate, err = t[keep], rate[keep], err[keep]

        if norms is not None:
            rate /= norms[i]

        line, poly = pu.errorpoly(t, rate, err, ealpha=0.2, ax=ax)
        lines.append(line)
        labels.append(label)

    ax.legend(lines, labels, loc='best', fontsize='small')


def specSnapshot(star, inst, trange, wrange, n=100, ax=None, vCen=None, maxpts=500, **kwargs):
    if ax is None: ax = plt.gca()
    ph, p = io.readphotons(star, inst)
    keep = mnp.inranges(p['time'], trange)
    p = p[keep]

    gtis = np.array([ph['gti'].data['start'], ph['gti'].data['stop']]).T
    gt = mnp.range_intersect([trange], gtis)
    dt = np.sum(gt[:,1] - gt[:,0])

    w0, w1, spec, err = sp.smooth_spec(p['wavelength'], p['epera'], n)
    spec, err = spec/dt, err/dt
    w = (w0 + w1) / 2.0

    keep, = np.nonzero(mnp.inranges(w, wrange))
    keep = np.insert(keep, [0, len(keep)], [keep[0]-1, keep[-1]+1])
    w, spec, err = w[keep], spec[keep], err[keep]

    if len(w) > maxpts:
        keep = un.downsample_even(w, maxpts)
        w, spec, err = w[keep], spec[keep], err[keep]

    velocify = lambda w: (w - vCen) / vCen * 3e5
    if vCen is not None:
        w = velocify(w)
        ax.set_xlabel('Doppler Velocity [km s$^{-1}$]')
        ax.set_xlim(map(velocify, wrange))
    else:
        ax.set_xlabel('Wavelength [$\AA$]')
        ax.set_xlim(wrange)
    ax.set_ylabel(fluxlabel)

    return pu.errorpoly(w, spec, err, **kwargs)