# -*- coding: utf-8 -*-
"""
A collection of functions for visually inspecting the data and data products.

Created on Wed Dec 10 15:22:01 2014

@author: Parke
"""
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import database as db
import io, settings, utils
import reduce as red
from plot import specstep
import numpy as np
from os import path
from spectralPhoton import image
from math import ceil, floor
from mypy.my_numpy import lace

stsfac = 2

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

def HSTcountregions(specfile, scale='auto'):
    """
    Show where the spectrum was extracted in a 2d histogram of counts created
    from the tag or corrtag file of the same name.
    """
    def ribbonstuff(seg):
        #get extraction region dimensions
        args = __ribbons(specfile, seg)
        N = args.pop()
        if '_sts_' in specfile:
            x = np.arange(1, N+1) * stsfac
        else:
            x = [1, N+1]
        args.append(x)
        __plotribbons(*args)
        return N / 2.0, args[0] # spec mid loc

    if '_sts_' in specfile:
        #read data
        tagfile = specfile.replace('x1d', 'tag')
        td = fits.getdata(tagfile, 1)

        #make image
        __cnts2img(td['axis1'], td['axis2'], scale)
        ribbonstuff('')

    if 'g130m' in specfile or 'g160m' in specfile:
        for seg in ['a', 'b']:
            #read data
            tagfile = specfile.replace('x1d', 'corrtag_'+seg)
            try:
                td = fits.getdata(tagfile, 1)
            except IOError:
                if seg == 'b':
                    print 'segment {} tag file not found'.format(seg)
                    continue
                else:
                    raise IOError()

            #create image
            plt.figure()
            __cnts2img(td['xcorr'], td['ycorr'], scale)
            ribbonstuff(seg)

    if 'cos_g230l' in specfile:
        #read data
        tagfile = specfile.replace('x1d', 'corrtag')
        td = fits.getdata(tagfile, 1)

        #create image
        plt.figure()
        __cnts2img(td['xcorr'], td['ycorr'], scale)

        for seg in ['a', 'b', 'c']:
            xmid, ymid = ribbonstuff(seg)
            plt.text(xmid, ymid, seg.upper(), {'fontsize':20}, ha='center',
                     va='center')

def piecespec(spec, err=True):
    """Plot a spectrum color-coded by source instrument."""
    inst = spec['instrument']
    for i in np.unique(inst):
        keep = (inst == i)
        thisspec = spec[keep]

        configs_i = np.nonzero(np.array(settings.instvals) & i)[0]
        configs = [settings.instruments[j] for j in configs_i]
        configstr = ' + '.join(configs)

        lines = specstep(thisspec, err=err)
        color = lines[0].get_color() if err else lines.get_color()
        w, f = [np.nanmean(thisspec[s]) for s in ['w0', 'flux']]
        txtprops = dict(facecolor='white', alpha=0.5, color=color)
        plt.text(w, f, configstr, bbox=txtprops, ha='center')

def vetcoadd(star, config):
    """Plot the components of a coadded spectrum to check that the coadd agrees."""
    coaddfile = db.coaddfile(star, config)
    coadd = io.read(coaddfile)
    assert len(coadd) == 1
    coadd = coadd[0]

    sourcefiles = coadd.meta['SOURCESPECS']
    sourcespecs = io.read(sourcefiles)
    for spec in sourcespecs:
        specstep(spec)

    specstep(coadd, lw=2.0, c='k', alpha=0.5)
    plt.title(path.basename(coadd.meta['FILENAME']))

def vetpanspec(pan_or_star, constant_dw=None, redlim=8000.0):
    """Plot unnormalized components of the panspec with the panspec to see that
    all choices were good. Phoenix spectrum is excluded because it is too big."""
    if type(pan_or_star) is str:
        star = pan_or_star
        panspec = io.read(db.panpath(star))[0]
    else:
        panspec = pan_or_star
        star = panspec.meta['STAR']
    files = db.panfiles(star)[0]
    panspec = utils.keepranges(panspec, 0.0, redlim)

    if constant_dw is None:
        plotfun = specstep
    else:
        panspec = red.evenbin(panspec, constant_dw)
        wbins = utils.wbins(panspec)
        def plotfun(spec, **kwargs):
            s = red.rebin(spec, wbins)
            return specstep(s, **kwargs)

    for f in files:
        if 'phx' in f: continue
        specs = io.read(f)
        for spec in specs:
            p = plotfun(spec, err=True)[0]
            x = (spec['w0'][0] + spec['w0'][-1])/2.0
            y = np.mean(spec['flux'])
            inst = db.parse_instrument(f)
            plt.text(x, y, inst,
                     bbox={'facecolor':'w', 'alpha':0.5, 'color':p.get_color()},
                     ha='center', va='center')
    plotfun(panspec, color='k', alpha=0.5)
    ymax = np.max(utils.keepranges(panspec, 3000.0, 8000.0)['flux'])
    plt.gca().set_ylim(-0.01 * ymax, 1.05 * ymax)
    plt.draw()

def vetnormfacs(spec, panspec, normfac, normranges):
    """Check the normalization of files by plotting normalized and unnormalieze
    versions. Called from reduce.panspec"""
    name = spec.meta['NAME']
    if normranges is None:
        normranges = utils.gapless_ranges(spec)

    # construct bins from lower res in each normrange
    overbins = []
    for normrange in normranges:
        getbins = lambda s: utils.wbins(utils.keepranges(s, normrange))
        possible_bins = map(getbins, [spec, panspec])
        bins = min(possible_bins, key=lambda b: len(b))
        overbins.append(bins)
    overbins = np.vstack(overbins)

    # rebin each spec to the lowest res within the overlap
    def coarsebin(s):
        overspec = red.rebin(s, overbins)
        return red.splice(s, overspec)
    spec, panspec = map(coarsebin, [spec, panspec])

    # plot the spectrum being normalized, highlighting the normranges
    plt.figure()
#    ax = plt.axes()
    inranges, _ = utils.argoverlap(spec, normranges)
    specin = spec[inranges]
    specout = spec[~inranges]
    kwargs = {'err' : False}
    line1 = specstep(specout, alpha=0.3, linestyle='--', **kwargs)
    kwargs['color'] = line1.get_color()
    line2 = specstep(specin, alpha=0.3, **kwargs)
    specin['flux'] *= normfac
    specout['flux'] *= normfac
    line3 = specstep(specout, linestyle='--', **kwargs)
    line4 = specstep(specin, **kwargs)
    plt.legend((line2, line4), ('original', 'normalized'))

    # plot panspec
    fullrange = spec['w0'][0], spec['w1'][-1]
    panspec = utils.keepranges(panspec, fullrange)
#    ymax = np.max(panspec['flux'])
    piecespec(panspec, err=False)
    normed = panspec['normfac'] != 1.0
    if np.any(normed):
        normspec = panspec[normed]
        normspec['flux'] /= normspec['normfac']
        specstep(normspec, color='k', linestyle='--', alpha=0.5, err=False)

#    ax.set_ylim(-0.05 * ymax, 1.05 * ymax)
    plt.title('{} normalization'.format(name))

def examinedates(star):
    """Plot the min and max obs dates to make sure everything looks peachy."""
    pans = io.readpans(star)
    mindate = min([np.min(p['minobsdate'][p['minobsdate'] > 0]) for p in pans])
    maxdate = max([np.max(p['maxobsdate'][p['maxobsdate'] > 0]) for p in pans])
    d = (maxdate - mindate)

    los = [specstep(p, key='minobsdate', linestyle='--') for p in pans]
    colors = [l.get_color() for l in los]
    his = [specstep(p, key='maxobsdate', linestyle=':', color=c) for p,c in
           zip(pans, colors)]
    plt.ylim(mindate - d/10.0, maxdate + d/10.0)
    labels = [db.parse_paninfo(p.meta['FILENAME']) for p in pans]
    plt.legend(los, labels)

def HSTimgregions(specfile):
    """
    Show where the spectrum was extracted from the corresponding STScI image
    files. Custom extractions made from an x2d ro sx2 will use those,
    stsci extractions use the crj or sfl since these are in pixel coordinates.
    """
    #find 2d image file
    pieces = specfile.split('_')
    custom = 'custom_spec' in specfile
    clip = 2 if custom else 1
    newfile = lambda suffix: '_'.join(pieces[:-clip] + [suffix])
    if custom:
        imgfiles = map(newfile, ['x2d.fits', 'sx2.fits'])
    else:
        imgfiles = map(newfile, ['crj.fits', 'sfl.fits', 'flt.fits'])
    imgfile = filter(path.exists, imgfiles)
    assert len(imgfile) == 1
    imgfile = imgfile[0]

    #plot 2d image file
    a = 0.3
    img = fits.getdata(imgfile, 1)
    m, n = img.shape
    plt.imshow(img**a, cmap='Greys')
    plt.gca().set_aspect('auto')
    plt.colorbar(label='flux**{:.2f}'.format(a))
    plt.ylabel('axis 1 (image)')
    plt.xlabel('axis 2 (wavelength)')

    if custom:
        spec = Table.read(specfile)
        smid, shgt, bhgt, bkoff = [spec.meta[s.upper()] for s in
            ['traceloc','extrsize','bksize', 'bkoff']]
        b1mid, b2mid = smid - bkoff, smid + bkoff
        b1hgt, b2hgt = bhgt, bhgt
        ribdims = [smid, shgt, b1mid, b1hgt, b2mid, b2hgt]
        x = [0, n+1]
    else:
        ribdims = __ribbons(specfile)
        if '_sts_' in specfile:
            N = ribdims.pop()
            x = np.arange(1, N+1)
            ribdims = [r/stsfac for r in ribdims]

    args = ribdims + [x]
    __plotribbons(*args)

def __ribbons(specfile, seg=''):
    if 'sts' in specfile:
        sd = fits.getdata(specfile, 1)
        smid = sd['extrlocy']*stsfac
        M, N = smid.shape
        b1mid, b2mid = [smid + sd[s][:, np.newaxis]*stsfac for s in
                        ['bk1offst', 'bk2offst']]
        shgt, b1hgt, b2hgt = [np.outer(sd[s], np.ones(N))*stsfac for s in
                        ['extrsize', 'bk1size','bk2size']]
        return [smid, shgt, b1mid, b1hgt, b2mid, b2hgt, N]
    if 'cos' in specfile:
        sh = fits.getheader(specfile, 1)
        smid = sh['sp_loc_'+seg]
        shgt = sh['sp_hgt_'+seg]
        b1mid, b2mid = sh['b_bkg1_'+seg], sh['b_bkg2_'+seg]
        b1hgt, b2hgt = sh['b_hgt1_'+seg], sh['b_hgt2_'+seg]
        N = sh['talen2']
        return [smid, shgt, b1mid, b1hgt, b2mid, b2hgt, N]

def __cnts2img(x,y, scalefunc):
    minx, maxx = floor(np.min(x)), ceil(np.max(x))
    miny, maxy = floor(np.min(y)), ceil(np.max(y))
    xbins, ybins = np.arange(minx, maxx+1), np.arange(miny, maxy+1)
    image(x, y, bins=[xbins, ybins], scalefunc=scalefunc, cmap='Greys')

def __plotribbons(smid, shgt, b1mid, b1hgt, b2mid, b2hgt, x):
        triplets = [[smid, shgt, 'g'], [b1mid, b1hgt, 'r'], [b2mid, b2hgt, 'r']]
        for m, h, c in triplets:
            __plotribbon(m, h, c, x)

def __plotribbon(mid, hgt, color, x):
    #get limits
    lo, hi = mid - hgt/2.0, mid + hgt/2.0

    #lace if mid and hgt aren't scalar
    if not np.isscalar(mid):
        x = lace(x, x[1:-1])
        lo = lace(lo, lo[:, 1:-1], 1)
        hi = lace(hi, hi[:, 1:-1], 1)
    else:
        lo, hi = [lo], [hi]

    for llo, hhi in zip(lo, hi):
        plt.fill_between(x, hhi, llo, color=color, alpha=0.5)