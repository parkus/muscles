# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
import os
import mypy.my_numpy as mnp
from math import sqrt, ceil, log10
from mypy import specutils, statsutils
import rc, utils, io, check, db
from spectralPhoton.hst.convenience import x2dspec, specphotons
import spectralPhoton.functions as sp
from itertools import combinations_with_replacement as combos
from scipy.stats import norm
from warnings import warn
import scicatalog.scicatalog as sc

colnames = rc.spectbl_format['colnames']
airglow_ranges = rc.airglow_ranges
safe_ranges = [0.0] + list(airglow_ranges.ravel()) + [np.inf]
safe_ranges = np.reshape(safe_ranges, [len(airglow_ranges) + 1, 2])


def theworks(star, R=10000.0, dw=1.0, silent=False):
    if np.isnan(rc.starprops['Teff'][star]):
        raise ValueError("Fool! You haven't entered Teff, etc. for {} yet."
                         "".format(star))

    try:
        rc.loadsettings(star)
    except IOError:
        warn("No settings file found for {}. Initializing one.".format(star))
        sets = rc.StarSettings(star)
        sets.save()

    # interpolate and save phoenix spectrum
    if not silent: print '\n\ninterpolating phoenix spectrum'
    auto_phxspec(star)

    # make custom extractions
    if not silent: print '\n\nperforming any custom extractions'
    auto_customspec(star)

    # coadd spectra
    if not silent: print '\n\ncoadding spectra'
    auto_coadd(star)

    # make panspectrum
    if not silent: print '\n\nstitching spectra together'
    panspectrum(star, R=R, dw=dw)  # panspec and Rspec


def panspectrum(star, R=10000.0, dw=1.0, savespecs=True, plotnorms=True,
                silent=False):
    """
    Coadd and splice the provided spectra into one panchromatic spectrum
    sampled at the native resolutions and constant R.

    Overlapping spectra will be normalized with the assumptions that they are
    listed in order of descending quality.
    """
    sets = rc.loadsettings(star)
    files, lyafile = db.panfiles(star)

    # if there is a custom normalization order, reorder the files accordingly
    if len(sets.norm_order) > 0:
        for inst in sets.norm_order[::-1]:
            ifiles = filter(lambda s: inst in s, files)
            for f in ifiles: files.remove(f)
            for f in ifiles[::-1]: files.insert(0, f)
    specs = io.read(files)
    names = [s.meta['NAME'] for s in specs]

    # make sure all spectra are of the same star
    star = __same_star(specs)

    # make sure spectra are each from a single source
    for i, s in enumerate(specs):
        try:
            __same_instrument([s])
        except ValueError:
            raise ValueError('More than one instrument used in spectbl {}'.format(i))

    # carry out custom trims according to user-defined settings
    if not silent:
        print 'trimming spectra according to settings for {}'.format(star)
    for i in range(len(specs)):
        goodranges = sets.get_custom_range(names[i])
        if goodranges is not None:
            if not silent:
                print ('trimming spectra in {} to the ranges {}'
                       ''.format(names[i], goodranges))
            specs[i] = utils.keepranges(specs[i], goodranges, ends='tight')

    # remove airglow lines from COS
    if not silent:
        print '\n\tremoving airglow from G130M spectrum'
    for i in range(len(specs)):
        if 'cos_g130m' in names[i]:
            specs[i] = utils.keepranges(specs[i], safe_ranges)
            # CLOOGE: remove some of g140m or e140m so it isn't used from 1198-lya
        #        if 'sts_e140m' in names[i] or 'sts_g140m' in names[i]:
        #            keep = [[0.0, safe_ranges[3,0]], [safe_ranges[3,1], np.inf]]
        #            specs[i] = utils.keepranges(specs[i], keep, ends='loose')

    # trim EUV and PHX models so they aren't used to fill small gaps in
    # UV data
    if not silent:
        print '\n\ttrimming PHOENIX and EUV model spectra'
    uvmin, uvmax = np.inf, 0.0
    for s, n in zip(specs, names):
        if db.parse_band(n) == 'u' and db.parse_observatory(n) != 'mod':
            thismin, thismax = s['w0'][0], s['w1'][-1]
            if thismin < uvmin: uvmin = thismin
            if thismax > uvmax: uvmax = thismax
    for i in range(len(specs)):
        if 'mod_phx' in names[i]:
            specs[i] = split_exact(specs[i], uvmax, 'red')
        if 'mod_euv' in names[i]:
            specs[i] = split_exact(specs[i], uvmin, 'blue')

    # normalize and splice according to input order
    spec = specs[0]
    if not silent:
        print '\nstarting stitched spectrum with {}'.format(spec.meta['NAME'])
    for i in range(1, len(specs)):
        addspec = specs[i]
        name = names[i]

        if not silent:
            specrange = [addspec['w0'][0], addspec['w1'][-1]]
            print ''
            print 'splicing in {}, covering {:.1f}-{:.1f}'.format(name, *specrange)

        if not rc.dontnormalize(addspec):
            inst = db.parse_instrument(name)
            if inst in sets.weird_norm:
                refinst = sets.weird_norm[inst]
                if not silent:
                    print 'normalizing {} spec using the same factor as that used for the {} spec'.format(inst, refinst)
                refspec = filter(lambda spec: refinst in spec.meta['NAME'], specs)
                assert len(refspec) == 1
                refspec = refspec[0]
                normfac = refspec[0]['normfac']
            else:
                overlap = utils.overlapping(spec, addspec)
                if not overlap and not silent:
                    print '\tno overlap, so won\'t normalize'
                if overlap:
                    if not silent:
                        print '\tnormalizing within the overlap'
                    normranges = sets.get_norm_range(name)
                    if normranges is None:
                        normspec = addspec
                    else:
                        normspec = utils.keepranges(addspec, normranges)
                    normfac = normalize(spec, normspec, silent=silent)
                    # HACK: phx plot breaks things, so I'm just not doing it for now
                if plotnorms and normfac != 1.0 and 'phx' not in name:
                    check.vetnormfacs(addspec, spec, normfac, normranges)
            addspec['flux'] *= normfac
            addspec['error'] *= normfac
            addspec['normfac'] = normfac
            specs[i] = addspec  # so i can use normalized specs later (lya)
        elif not silent:
            print '\twon\'t normalize, cuz you said not to'

        spec = smartsplice(spec, addspec, silent=silent)
    spec.meta['NAME'] = db.parse_name(db.panpath(star))

    # replace lya portion with model or normalized stis data
    if lyafile is None:
        name = filter(lambda s: 'sts_g140m' in s or 'sts_e140m' in s, names)
        if len(name) > 1:
            raise Exception('More than one Lya stis file found.')
        ilya = names.index(name[0])
        lyaspec = specs[ilya]
        normfac = lyaspec['normfac'][0]
        if not silent:
            print ('replacing section {:.1f}-{:.1f} with STIS data from {lf}, '
                   'normalized by {normfac}'
                   ''.format(*rc.lyacut, lf=lyafile, normfac=normfac))
    else:
        if not silent:
            print ('replacing section {:.1f}-{:.1f} with data from {lf}'
                   ''.format(*rc.lyacut, lf=lyafile))
        lyaspec = io.read(lyafile)[0]
    lyaspec = utils.keepranges(lyaspec, rc.lyacut)
    spec = splice(spec, lyaspec)

    # fill any remaining gaps
    order, span = 2, 20.0
    if not silent:
        print ('filling in any gaps with an order {} polynomial fit to an '
               'area {}x the gap width'.format(order, span))
    spec = fill_gaps(spec, fill_with=order, fit_span=span, silent=silent,
                     mingapR=10.0)

    # resample at constant R and dR
    if not silent:
        print ('creating resampled panspecs at R = {:.0f} and dw = {:.1f} AA'
               ''.format(R, dw))
    Rspec = powerbin(spec, R)
    dspec = evenbin(spec, dw)

    # save to fits
    if savespecs:
        paths = [db.panpath(star), db.Rpanpath(star, R), db.dpanpath(star, dw)]
        if not silent:
            print 'saving spectra to \n' + '\n\t'.join(paths)
        for s, path in zip([spec, Rspec, dspec], paths):
            io.writefits(s, path, overwrite=True)

    return spec, Rspec, dspec


def solarspec(date):
    """
    Read and splice together ultraviolet and visible solar spectra. These
    must be in the muscles/solar_data folder and appropriately named. This
    means if you want new ones you need to get them using LASP solar data
    portal thingy. Date format is yyyy-mm-dd.
    """
    ufile, vfile = db.solarfiles(date)
    u, v = [io.readcsv(f)[0] for f in (ufile, vfile)]
    w = 1850.0
    u = split_exact(u, w, 'blue')
    v = split_exact(v, w, 'red')
    return utils.vstack([u, v])


def normalize(spectbla, spectblb, worry=0.05, flagmask=False, silent=False):
    """
    Normalize the spectrum b to spectrum a.

    Parameters
    ----------
    spectbla : muscles spectrum
        spectrum to normalize against
    spectblb : muscles spectrum
        spectrum to be normalized. wavelength and flux units must be the
        same in a and b
    worry : {float|False}
        consider errors in the overlapping area of the spectra. if the
        areas are consistent with being identical with probability p
        > worry, normalization will not occur. If normalization does
        occur, an error in the normalization factor will be computed and
        propagated.
    flagmask : {True|False}
        Mask data that is flagged when computing areas of overlap.
    silent : {True|False}
        Print lots of warm fuzzies...

    Returns
    -------
    normspec : muscles spectrum
        normalized spectrum

    Notes
    -----
    - if spectbla has all 0.0 errors (meaning it's a model) normalization
        does not occur
    """

    if not silent:
        names = [s.meta['NAME'] for s in [spectbla, spectblb]]

    # parse out the overlap
    overa, overb = utils.argoverlap(spectbla, spectblb, 'tight')
    if np.sum(overa) == 0 or np.sum(overb) == 0:
        if not silent:
            print ('no full pixel overlap for at least one spectrum. can\'t '
                   'normalize.')
        return 1.0

    # if speca has all zero errors (it's a model), don't normalize to it
    if np.all(spectbla[overa]['error'] == 0.0):
        if not silent:
            print ('the master spectrum {} has all zero errors, so {} will '
                   'not be normalized to it'.format(*names))
            return 1.0
        #        return spectblb

    # rebin to the coarser spectrum
    if np.sum(overa) < np.sum(overb):
        ospeca = spectbla[overa]
        wbins = utils.wbins(ospeca)
        ospecb = rebin(spectblb, wbins)
        order = slice(None, None, -1)
    else:
        ospecb = spectblb[overb]
        wbins = utils.wbins(ospecb)
        ospeca = rebin(spectbla, wbins)
        order = slice(None, None, 1)
    if not silent:
        over_range = [ospeca['w0'][0], ospeca['w1'][-1]]
        print ('spectra overlap at {:.2f}-{:.2f}'.format(*over_range))
        print ('rebinning {} to the (coarser) resolution of {} where they '
               'overlap'.format(*names[order]))

    # mask data with flags
    mask = np.zeros(len(ospeca), bool)
    if flagmask:
        flagged = (ospeca['flags'] > 0) | (ospecb['flags'] > 0)
        mask[flagged] = True
        if not silent:
            percent_flagged = np.sum(flagged) / float(len(ospeca)) * 100.0
            print ('{:.2f}% of the data that was flagged in one spectra or '
                   'the other. masking it out.'.format(percent_flagged))

    # mask data where speca has 0.0 errors
    zeroerr = (ospeca['error'] == 0.0)
    mask[zeroerr] = True
    if not silent:
        percent_zeroerr = np.sum(zeroerr) / float(len(ospeca)) * 100.0
        print ('{:.2f}% of the data in the master spectrum had zero error. '
               'masking it out'.format(percent_zeroerr))

    good = ~mask

    # compute normalization factor
    ospecs = [ospeca, ospecb]
    dw = wbins[:, 1] - wbins[:, 0]

    def getarea(spec):
        area = np.sum(spec['flux'][good] * dw[good])
        error = mnp.quadsum(spec['error'][good] * dw[good])
        return area, error

    areas, errors = zip(*map(getarea, ospecs))
    if not silent:
        print ('master spectrum has overlap area    {:.2e} ({:.2e})\n'
               'secondary spectrum has overlap area {:.2e} ({:.2e})'
               ''.format(areas[0], errors[0], areas[1], errors[1]))
    diff = abs(areas[1] - areas[0])
    differr = mnp.quadsum(errors)
    p = 2.0 * (1.0 - norm.cdf(diff, loc=0.0, scale=differr))
    if not silent:
        print ('difference =                        {:.2e} ({:.2e})'
               ''.format(diff, differr))
        print 'probability that the difference is spurious = {:.4f}'.format(p)
    if worry and p > worry:
        if not silent:
            print ('{} > {}, so secondary will not be normalized to master'
                   ''.format(p, worry))
        return 1.0
    #        return spectblb
    normfac = areas[0] / areas[1]
    if worry:
        normfacerr = sqrt((errors[0] / areas[1]) ** 2 +
                          (areas[0] * errors[1] / areas[1] ** 2) ** 2)
    else:
        normfacerr = 0.0
    if not silent:
        print ('secondary will be normalized by a factor of {} ({})'
               ''.format(normfac, normfacerr))

    #    normspec = Table(spectblb, copy=True)
    #    nze = (normspec['error'] != 0.0)
    #    normspec['error'][nze] = mnp.quadsum([normspec['error'][nze]*normfac,
    #                                          normspec['flux'][nze]*normfacerr], axis=0)
    #    normspec['flux'] *= normfac
    #    normspec['normfac'] = normfac
    #    return normspec
    return normfac


def smartsplice(spectbla, spectblb, minsplice=0.005, silent=False):
    """
    Splice one spectrum into another (presumably overlapping) spectrum in a
    way that minimizes overall error.

    Chooses a contiguous region of the overlap between spetrum a and spectrum
    b such that the integrated error in the overlap is as low as possible.
    This could mean not splicing spectrum b in at all.

    Parameters
    ----------
    spectbla : astropy Table
        Table of the base spectrum. Must not contain any gaps.
    spectblb : astropy Table
        Table of the spectrum to be splice into spectbla.
    minsplice : float
        Minimum size of the splice relative to the central wavelength value
        of the overlapping range. Set to 0.0 in order to allow splices of any
        size. This is only valid for splices where the range of one spectrum
        is entirely within another.

    Returns
    -------
    splicedspec : astropy Table
        The spliced spectrum.
    """
    # sort the two spectra
    both = [spectbla, spectblb]
    key = lambda s: s['w0'][0]
    both.sort(key=key)
    spec0, spec1 = both
    if not silent:
        names = [s.meta['NAME'] for s in both]

    if not utils.overlapping(*both):  # they don't overlap
        specs = sum(map(utils.gapsplit, both), [])
        specs.sort(key=key)
        spec = utils.vstack(specs)
        assert np.all(spec['w0'][1:] > spec['w0'][:-1])
        return spec

    # get their overlap and the range of the overlap
    over0, over1 = utils.argoverlap(*both, method='loose')
    ospec0, ospec1 = spec0[over0], spec1[over1]
    oboth = [ospec0, ospec1]
    wr0, wr1 = [[s['w0'][0], s['w1'][-1]] for s in oboth]
    wr = [max(wr0[0], wr1[0]), min(wr0[1], wr1[1])]

    # if the spectra have gaps within the overlap, split them at their gaps,
    # sort them, and splice in pairs
    gaps0, gaps1 = map(utils.gapranges, both)
    gapsin0 = np.any(mnp.inranges(gaps0.flatten(), wr1))
    gapsin1 = np.any(mnp.inranges(gaps1.flatten(), wr0))
    if gapsin0 or gapsin1:
        specs = sum(map(utils.gapsplit, both), [])
        specs.sort(key=key)
        return reduce(smartsplice, specs)

    # if either spectrum has zeros for errors, don't use it for any of the
    # overlap
    allzeroerrs = lambda spec: np.all(spec['error'] == 0.0)

    # somehow the error for one of the phx entries is being changed to 0
    # from nan, so doing allnan on the original spectrum is a workaround
    ismodel0 = allzeroerrs(ospec0) or allzeroerrs(spec0)
    ismodel1 = allzeroerrs(ospec1) or allzeroerrs(spec1)
    if ismodel1 and not ismodel0:
        return splice(spec1, spec0)
    if ismodel0 and not ismodel1:
        return splice(spec0, spec1)
    if ismodel0 and ismodel1:
        return NotImplementedError('Not sure how to splice two models together.')

    # otherwise, find the best splice locations
    # get all edges within the overlap
    we0, we1 = [__edgesinrange(s, wr) for s in both]
    we = np.hstack([wr[0], we0, we1])
    we = np.unique(we)  # also sorts the array
    wbins = utils.edges2bins(we)
    # wr[1] is already included because of how searchsorted works

    # rebin spectral overlap to we
    oboth = [rebin(o, wbins) for o in oboth]
    dw = np.diff(we)

    # get flux and variance and mask values with dq flags and nan values
    masks = [(spec['flags'] > 0) for spec in oboth]
    flus = [spec['flux'] * dw for spec in oboth]
    sig2s = [(spec['error'] * dw) ** 2 for spec in oboth]

    def maskitfillit(x, mask):
        x[mask] = 0.0
        x[np.isnan(x)] = 0.0
        return x

    flus = map(maskitfillit, flus, masks)
    sig2s = map(maskitfillit, sig2s, masks)

    sumstuff = lambda x: np.insert(np.cumsum(x), 0, 0.0)
    cf0, cf1 = map(sumstuff, flus)
    cv0, cv1 = map(sumstuff, sig2s)
    # this way, any portions where all variances are zero will result in a
    # total on nan for the signal to noise

    enclosed = (wr1[1] < wr0[1])
    if enclosed:
        wmid = (wr[1] + wr[0]) / 2.0
        mindw = minsplice * wmid
        if len(we) < 500:
            indices = np.arange(len(we))
        else:
            indices = np.round(np.linspace(0, len(we) - 1, 500)).astype(int)
        ijs = combos(indices, 2)
        i, j = np.array([ij for ij in ijs]).T
        keep = (we[j] - we[i]) > mindw
        i, j = i[keep], j[keep]
        i, j = map(np.append, [i, j], [0, 0])

        #        signal = cf0[i] + (cf1[j] - cf1[i]) + (cf0[-1] - cf0[j])
        var = cv0[i] + (cv1[j] - cv1[i]) + (cv0[-1] - cv0[j])
        #        SN = signal/np.sqrt(var)
        # pick the best and splice the spectra
        #        best = np.nanargmax(SN)
        best = np.nanargmin(var)
        i, j = i[best], j[best]
        cut0, cut1 = we[i], we[j]
        if i == j:
            return spec0

        splicespec = spec1
        splicespec = split_exact(splicespec, cut0, 'red')
        splicespec = split_exact(splicespec, cut1, 'blue')
        if not silent:
            print ('spectrum {} spliced into {} from {:.2f} to {:.2f}'
                   ''.format(names[1], names[0], cut0, cut1))

    # do the same, if not enclosed
    else:
        i = range(len(we))
        #        signal = cf0[i] + (cf1[-1] - cf1[i])
        var = cv0[i] + (cv1[-1] - cv1[i])
        #        SN = signal/np.sqrt(var)
        #        best = np.nanargmax(SN)
        best = np.nanargmin(var)
        i = i[best]
        cut = we[best]
        splicespec = split_exact(spec1, cut, 'red')

        if not silent:
            print ('spectrum {} spliced into {} from {:.2f} onward'
                   ''.format(names[1], names[0], cut))

    spec = splice(spec0, splicespec)
    assert np.all(spec['w0'][1:] > spec['w0'][:-1])
    return spec


def splice(spectbla, spectblb):
    """
    Replace spectrum a with spectrum b where they overlap.

    The outer bins of spectrum b are preserved, whereas the bins adjacent
    to the edges of spectrum b in spectrum a may be cut off. If so, the errors
    for the fractional bins are appropriately augmented assuming Poisson
    statistics and a constant flux within the original bins.
    """
    # if spectrum b has gaps, divide it up and add the pieces it into spectbla
    # separately
    if utils.hasgaps(spectblb):
        bspecs = utils.gapsplit(spectblb)
        return reduce(splice, bspecs, spectbla)

    # cut up a according to the range of b and stack
    speclist = []
    leftspec = split_exact(spectbla, spectblb['w0'][0], 'blue')
    speclist.append(leftspec)
    speclist.append(spectblb)
    rightspec = split_exact(spectbla, spectblb['w1'][-1], 'red')
    speclist.append(rightspec)
    spec = utils.vstack(speclist)

    # modify metadata
    metas = [s.meta for s in [spectbla, spectblb]]

    def parsesources(meta):
        if len(meta['SOURCESPECS']):
            return meta['SOURCESPECS']
        else:
            return meta['NAME']

    sources = np.hstack(map(parsesources, metas))
    spec.meta['SOURCESPECS'] = np.unique(sources)
    spec.meta['FILENAME'] = ''
    spec.meta['NAME'] = 'stitched spectrum'

    assert np.all(spec['w0'][1:] > spec['w0'][:-1])
    return spec


def split_exact(spectbl, w, keepside):
    """
    Split a spectrum at exactly the specified wavelength, dealing with new
    fractional bins by augmenting the error according to Poisson statistics.

    Parameters
    ----------
    spectbl : muscles spectrum
    w : float
        wavelength at which to trim
    keepside : {'red'|'blue'|'both'}

    Result
    ------
    splitspecs : muscles spectrum
        one or two spectables according to the keepside setting
    """
    keepblu = (keepside in ['blue', 'both'])
    keepred = (keepside in ['red', 'both'])

    # find the index of the bin w falls in
    flag, i = utils.specwhere(spectbl, w)

    if flag == 1:
        # w is in a bin of the spectbl
        # parse out info from bin that covers w
        error = spectbl[i]['error']
        w0, w1 = spectbl['w0'][i], spectbl['w1'][i]
        dw = w1 - w0

        # make tables with modified edge bin
        if keepblu:
            if w == w0:
                bluspec = Table(spectbl[:i], copy=True)
            else:
                bluspec = Table(spectbl[:i + 1], copy=True)
                dw_new = w - w0
                error_new = error * sqrt(dw / dw_new)
                bluspec[-1]['w1'] = w
                bluspec[-1]['error'] = error_new
        if keepred:
            redspec = Table(spectbl[i:], copy=True)
            if w != w0:
                dw_new = w1 - w
                error_new = error * sqrt(dw / dw_new)
                redspec[0]['w0'] = w
                redspec[0]['error'] = error_new
    else:
        # w is outside of the spectbl, in a gap, or right on a bin edge,
        # then i works as a slice
        if keepblu:
            bluspec = Table(spectbl[:i], copy=True)
        if keepred:
            redspec = Table(spectbl[i:], copy=True)

    if keepblu: assert np.all(bluspec['w1'] > bluspec['w0'])
    if keepred: assert np.all(redspec['w1'] > redspec['w0'])

    if keepside == 'blue':
        return bluspec
    if keepside == 'red':
        return redspec
    if keepside == 'both':
        return bluspec, redspec


def powerbin(spectbl, R=1000.0, lo=1.0, hi=None):
    """
    Rebin a spectrum onto a grid with constant resolving power.

    If the constant R grid cannot does not permit an integer number of bins
    within the original wavelength range, the remainder will be discarded.
    """
    start = spectbl['w0'][0]
    if start < lo: start = lo
    end = spectbl['w1'][-1] if hi is None else hi
    fac = (2.0 * R + 1.0) / (2.0 * R - 1.0)
    maxpow = ceil(log10(end / start) / log10(fac))
    powers = np.arange(maxpow)
    we = start * fac ** powers
    wbins = utils.edges2bins(we)
    return rebin(spectbl, wbins)


def evenbin(spectbl, dw, lo=None, hi=None):
    if lo is None: lo = np.min(spectbl['w0'])
    if hi is None: hi = np.max(spectbl['w1'])
    newedges = np.arange(lo, hi, dw)
    newbins = utils.edges2bins(newedges)
    return rebin(spectbl, newbins)


def coadd(spectbls, maskbaddata=True, savefits=False, weights='exptime',
          silent=False):
    """Coadd spectra in spectbls. weights can be 'exptime' or 'error'"""
    inst = __same_instrument(spectbls)
    # star = __same_star(spectbls)
    star = spectbls[0].meta['STAR']

    sourcefiles = [s.meta['FILENAME'] for s in spectbls]

    listify = lambda s: [spec[s].data for spec in spectbls]
    w0, w1, f, e, expt, dq, inst, normfac, start, end = map(listify, colnames)
    we = [np.append(ww0, ww1[-1]) for ww0, ww1 in zip(w0, w1)]

    if any([np.any(n != 1.0) for n in normfac]):
        warn("Spectra with normfacs != 1.0 are being cladded.")

    weights = [1.0 / ee ** 2 for ee in e] if weights == 'error' else expt
    if maskbaddata:
        dqmasks = map(rc.seriousdqs, sourcefiles)
        masks = __make_masks(we, dq, dqmasks)
        for i in range(len(masks)):
            start[i][masks[i]] = np.inf
            end[i][masks[i]] = -np.inf
            inst[i][masks[i]] = 0
        cwe, cf, ce, cexpt, dq = specutils.coadd(we, f, e, weights, dq, masks)
    else:
        cwe, cf, ce, cexpt, dq = specutils.coadd(we, f, e, weights, dq)

    data = [inst, start, end]
    funcs = ['or', 'min', 'max']
    basevals = [0, np.inf, -np.inf]

    def specialcoadder(a, f, bv):
        return specutils.stack_special(we, a, f, commongrid=cwe, baseval=bv)

    cinst, cstart, cend = map(specialcoadder, data, funcs, basevals)
    cnorm = np.ones(len(cwe) - 1)
    cw0, cw1 = cwe[:-1], cwe[1:]

    goodbins = (cexpt > 0)
    data = [v[goodbins] for v in [cw0, cw1, cf, ce, cexpt, dq, cinst, cnorm, cstart, cend]]
    cfile = db.coaddpath(sourcefiles[0])
    cname = db.parse_name(cfile)
    sourcespecs = list(set([s.meta['NAME'] for s in spectbls]))
    spectbl = utils.list2spectbl(data, star, name=cname, sourcespecs=sourcespecs)

    if all([np.all(s['instrument'] > 0) for s in spectbls]):
        assert np.all(spectbl['instrument'] > 0)
    assert not np.any(spectbl['minobsdate'] < 0)
    if np.all(spectbl['minobsdate'] > 0):
        assert np.all(spectbl['maxobsdate'] > spectbl['minobsdate'])
    else:
        assert np.all(spectbl['maxobsdate'] >= spectbl['minobsdate'])

    if savefits:
        io.writefits(spectbl, cfile, overwrite=True)
        spectbl.meta['FILENAME'] = cfile
        if not silent: print 'coadd saved to \n\t{}'.format(cfile)
    return spectbl


def auto_coadd(star, configs=None, silent=False):
    if configs is None:
        groups = db.coaddgroups(star)
    else:
        if type(configs) is str: configs = [configs]
        groups = []
        for config in configs:
            files = db.sourcespecfiles(star, config)
            files = db.sub_customfiles(files)
            # clooge to deal with case when there are no x1ds -- look for just custom specs
            if len(files) == 0:
                files = db.findfiles('u', star, config, 'custom_spec', fullpaths=True)
            groups.append(files)

    for group in groups:
        spectbls = sum(map(io.read, group), [])
        if len(spectbls) == 1:
            if not silent:
                print ('single spectrum for {}, moving on'
                       ''.format(spectbls[0].meta['NAME']))
            continue
        if not silent:
            names = [s.meta['NAME'] for s in spectbls]
            print 'coadding the spectra \n\t{}'.format('\n\t'.join(names))
        echelles = map(utils.isechelle, group)
        if any(echelles):
            weights = 'error'
            if not silent:
                print 'some files are echelles, so weighting by 1/error**2'
        else:
            weights = 'exptime'
            if not silent:
                print 'weighting by exposure time'

        if len(spectbls) > 1:
            coadd(spectbls, savefits=True, weights=weights, silent=silent)


def phxspec(Teff, logg=4.5, FeH=0.0, aM=0.0, repo=rc.phxrepo):
    """
    Quad-linearly interpolates the available phoenix spectra to the provided
    values for temperature, surface gravity, metallicity, and alpha metal
    content.
    """
    grids = [rc.phxTgrid, rc.phxggrid, rc.phxZgrid, rc.phxagrid]
    pt = [Teff, logg, FeH, aM]

    # make a function to retrieve spectrum given grid indices
    def getspec(*indices):
        args = [grid[i] for grid, i in zip(grids, indices)]
        return io.phxdata(*args, repo=repo)

    # interpolate
    spec = mnp.sliminterpN(pt, grids, getspec)

    # make spectbl
    N = len(spec)
    err = np.zeros(N)
    expt, flags = np.zeros(N), np.zeros(N, 'i1')
    insti = rc.getinsti('mod_phx_-----')
    source = insti * np.ones(N, 'i1')
    normfac, start, end = 1.0, 0.0, 0.0
    data = [rc.phxwave[:-1], rc.phxwave[1:], spec, err, expt, flags, source,
            normfac, start, end]
    return utils.list2spectbl(data)


def auto_phxspec(star, silent=False):
    kwds = {}
    for key in ['Teff', 'logg', 'FeH', 'aM']:
        val = rc.starprops[key][star]
        if not np.isnan(val):
            kwds[key] = val
    if not silent:
        print 'interpolating phoenix spectrum for {} with values'.format(star)
        print kwds
    spec = phxspec(**kwds)
    spec.meta['STAR'] = star
    path = rc.phxpath(star)
    spec.meta['NAME'] = db.parse_name(path)
    if not silent:
        print 'writing spectrum to {}'.format(path)
    io.writefits(spec, path, overwrite=True)


def auto_customspec(star, silent=False):
    ss = rc.loadsettings(star)
    if not silent:
        if len(ss.custom_extractions) == 0:
            'no custom extractions set for {}'.format(star)
    for custom in ss.custom_extractions:
        config = custom['config']
        if not silent:
            print 'custom extracting spectrum for {}'.format(config)
            print 'with parameters'
            print custom['kwds']
        if 'hst' in config:
            x2dfiles = db.findfiles('u', star, config, 'x2d', fullpaths=True)
            for x2dfile in x2dfiles:
                if not silent:
                    print 'using x2dfile {}'.format(x2dfile)
                x1dfile = x2dfile.replace('x2d', 'x1d')
                if not os.path.exists(x1dfile): x1dfile = None
                specfile = x2dfile.replace('x2d', 'custom_spec')
                dqmask = rc.seriousdqs(specfile)
                spec = x2dspec(x2dfile, x1dfile=x1dfile, bkmask=dqmask, **custom['kwds'])

                # trim any nans
                isnan = np.isnan(spec['flux'])
                spec = spec[~isnan]

                # conform to spectbl standard
                meta = spec.meta
                datalist = [spec[s] for s in ['w0', 'w1', 'flux', 'error']]
                hdr = fits.getheader(x2dfile, extname='sci')
                expt, start, end = [hdr[s] for s in ['exptime', 'expstart', 'expend']]
                inst = rc.getinsti(specfile)
                norm = 1.0
                datalist.extend([expt, spec['dq'], inst, norm, start, end])

                spectbl = utils.list2spectbl(datalist, star, specfile, '', [x2dfile])
                for key in meta:
                    spectbl.meta[key] = meta[key]

                if not silent:
                    print 'saving custom extraction to {}'.format(specfile)
                io.writefits(spectbl, specfile, overwrite=True)
        else:
            raise NotImplementedError("No custom extractions defined for {}"
                                      "".format(config))


def auto_photons(star, inst='all'):
    alltagfiles = db.findfiles('u', 'tag', star, fullpaths=True)
    allx1dfiles = db.findfiles('u', 'x1d', star, fullpaths=True)

    if inst == 'all':
        instruments = map(db.parse_instrument, alltagfiles)
        instruments = list(set(instruments))
        #FIXME: some echelle data have different numbers of orders, which the function for finding overlapping
        instruments = filter(lambda s: 'cos' in s, instruments)
    else:
        instruments = [inst]

    for instrument in instruments:
        getInstFiles = lambda files: filter(lambda s: instrument in s, files)
        tagfiles = getInstFiles(alltagfiles)
        x1dfiles = getInstFiles(allx1dfiles)

        if 'cos_g230l' in instrument:
            kwds = {'extrsize':30, 'bkoff':[30, -30], 'bksize':[20, 20]}
        else:
            kwds = {}
        f = db.photonpath(tagfiles[0])
        specphotons(tagfiles, x1dfiles, fitsout=f, clobber=True, **kwds)


def rebin(spec, newbins):
    """Rebin the spectrum, dealing with gaps in newbins if appropriate."""

    # get overlapping bins, warn if some don't overlap
    _, overnew = utils.argoverlap(spec, newbins, method='tight')
    Nkeep = np.sum(overnew)
    if Nkeep == 0:
        warn('All newbins fall outside of spec. Returning empty spectrum.')
        return spec[0:0]
    if Nkeep < len(newbins):
        warn('Some newbins fall outside of spec and will be discarded.')
    newbins = newbins[overnew]

    # split at gaps and rebin. no bins covering a gap in spec should remain in
    # newbins, so there shouldn't be a need to split newgaps
    splitbins = utils.gapsplit(newbins)
    if len(splitbins) > 1:
        specs = []
        for bins in splitbins:
            trim = utils.keepranges(spec, bins[0, 0], bins[-1, 1], ends='loose')
            specs.append(rebin(trim, bins))
        return utils.vstack(specs)

    # trim down spec to avoid gaps (gaps are handled in code block above)
    spec = utils.keepranges(spec, newbins[0, 0], newbins[-1, 1], ends='loose')

    # rebin
    w0, w1 = newbins.T
    newedges = utils.bins2edges(newbins)
    oldedges = utils.wedges(spec)
    dwnew, dwold = map(np.diff, [newedges, oldedges])
    flux, error, flags = specutils.rebin(newedges, oldedges, spec['flux'],
                                         spec['error'], spec['flags'])
    insts = mnp.rebin(newedges, oldedges, spec['instrument'], 'or')
    normfac = mnp.rebin(newedges, oldedges, spec['normfac'], 'avg')
    start = mnp.rebin(newedges, oldedges, spec['minobsdate'], 'min')
    end = mnp.rebin(newedges, oldedges, spec['maxobsdate'], 'max')
    expt = mnp.rebin(newedges, oldedges, spec['exptime'], 'avg')

    # spectbl accoutrments
    star, name, fn, sf = [spec.meta[s] for s in
                          ['STAR', 'NAME', 'FILENAME', 'SOURCESPECS']]

    return utils.vecs2spectbl(w0, w1, flux, error, expt, flags, insts, normfac,
                              start, end, star, fn, name, sf)


def remove_line(spec, wavelength, fill_with=None, minclip=None, silent=False):
    """
    Remove a line from a spectrum by automatically identifying its extent and
    clipping it out.

    Parameters
    ----------
    spec : muscles spectbl
    wavelength : float
        central wavelength of line
    fill_with : {None|int}
        Whether to fill the resulting gap. If None, just return spectrum
        with a gap. If an integer, fit a polynomial of that order and use
        it to fill the gap.
    minclip : float, optional
        Minumum wavlength range to clip out of spectrum relative to the
        line wavelength. E.g. 0.001 means clip out at elast 0.001 * wavelength
        if no line is identified.

    Returns
    -------
    spec : muscles spectbl
        spectrum with line removed or filled, if found
    """
    w = wavelength
    n = 4 if fill_with is None else fill_with

    def fallback_range():
        radius = w / minclip / 2.0
        if not silent:
            print ('using minclip range of +- {} AA (R = {})'
                   ''.format(radius, minclip))
        return [w - radius, w + radius]

    # clip spectrum down to just a window around the line for fast processing
    width = w * 0.01
    window = [w - width / 2.0, w + width / 2.0]
    argmini = utils.argrange(spec, window)

    # exclude any zero-error values. polyfit can't handle 'em
    argmini[spec['error'] <= 0.0] = False
    minispec = spec[argmini]

    # identify lines vs continuum in spectrum
    wbins = utils.wbins(minispec)
    flux, error = minispec['flux'], minispec['error']
    pcont, pline = 1.0 - 1e-5, 1.0 - 1e-10
    try:
        flags = specutils.split(wbins, flux, error, contcut=pcont,
                                linecut=pline, contfit=n, silent=silent)

        # record the continuum points in spec that should be fit
        cont = (flags == 3)
        fit_pts = np.nonzero(argmini)[0]
        fit_pts = fit_pts[cont]

        # isolate range of emission feature that includes wavelength
        line_ranges = specutils.flags2ranges(wbins, flags == 1)
        inrange = (w <= line_ranges[:, 1]) & (w >= line_ranges[:, 0])

        # if an emission feature does include it, get its range
        if np.any(inrange):
            if not silent:
                print 'successfully located line at {} AA'.format(w)
            bad_range = np.squeeze(line_ranges[inrange, :])
        else:
            if not silent:
                print ('none of the identified lines covered the provided '
                       'wavelength of {:.2f} AA'.format(w))
            if minclip is None:
                if not silent:
                    print 'returning unchanged spectrum'
                return spec
            else:
                bad_range = fallback_range()
    except ValueError:
        if not silent:
            print ('could not separate line from continuum emission within '
                   '{:.2f}-{:.2f}'.format(*window))
        if minclip is None:
            if not silent:
                print 'returning unchanged spectrum'
            return spec
        else:
            bad_range = fallback_range()
            fit_pts = argmini

    gap = utils.argrange(spec, bad_range)
    comment = 'clipped out data from {:.2f}-{:.2f} AA'.format(*bad_range)
    if not silent:
        print comment
    spec.meta['COMMENT'].append(comment)

    # fill the gap, if desired
    if fill_with is None:
        spec = spec[~gap]
        if not silent:
            print 'leaving a gap where data was clipped out'
    else:
        # get the bins to be filled
        gapbins = utils.wbins(spec[gap])

        # fill 'em
        spec = fill_gaps(spec, n, fit_pts=fit_pts, gapbins=gapbins,
                         silent=silent)

    return spec


def fill_gaps(spec, fill_with=4, fit_span=10.0, fit_pts=None, resolution=None,
              mingapR=10.0, gapbins=None, silent=False):
    """
    Fill gaps in the spectrum with a polynomial fit to the continuum in an
    area encompassing the gap.

    Parameters
    ----------
    spec : muscles spectbl
    fill_with : int, optional
        Order of the polynomial fit to use for filling the gap
    fit_span : float, optional
        Width of the area centered on the gap within which to fit the
        polynomial, in multiples of the gap width.
    fit_pts : 1D array
        Boolean array the same length as spec that specifies which points
        to use for fitting the polynomial. Overrides fit_span.
    resolution : float, optional
        The resolution at which to bin the polynomial for filling the gap. If
        None, the average resolution in fit_span or fit_pts is used.
    gapbins : 2D array
        Specific bins to use when filling a single known gap. Overrides
        resolution. The bins are not checked to see if they are flush with the
        gap borders.
    silent : {True|False}, optional

    Returns
    -------
    filledspec : muscles spectbl
        Spectrum with the gap filled.
    """
    findgaps = (gapbins is None)
    if findgaps and not utils.hasgaps(spec):
        if not silent:
            print 'No gaps in spectrum.'
        return spec

    wbins = utils.wbins(spec)
    flux, error = spec['flux'], spec['error']
    n = fill_with

    # if fit_pts provided, go ahead and fit polynomial to those
    if fit_pts is not None:
        poly = specutils.polyfit(wbins[fit_pts, :], flux[fit_pts], n,
                                 error[fit_pts])[2]

    # identify gaps
    if findgaps:
        gapranges = utils.gapranges(spec)
    else:
        gapranges = np.array([[gapbins[0, 0], gapbins[-1, 1]]])

    gapspecs = []
    comments = []
    for gr in gapranges:
        width = gr[1] - gr[0]
        midpt = (gr[0] + gr[1]) / 2.0
        gapR = midpt / width
        if gapR < mingapR:
            if not silent:
                print ('gap from {:.2f}-{:.2f} has R = {:.1f} < {:.1f} = '
                       'mingap, skipping'.format(gr[0], gr[1], gapR, mingapR))
            gapspecs.append(spec[0:0])
            continue
        if fit_pts is None:
            # confine spectrum to just fit_span around gap
            radius = (fit_span * width) / 2.0
            wspan = [midpt - radius, midpt + radius]
            span = utils.argrange(spec, wspan)

            if np.sum(span) < 100:
                i = utils.specwhere(spec, midpt)[1]
                span[i - 50:i + 50] = True

            if not silent:
                print 'fitting polynomial to range {:.2f}-{:.2f}'.format(*wspan)
                print 'attempting to mask out spectral lines'

            # also exclude zero-error (model) points
            span[spec['error'] <= 0.0] = False
            # and points from a different instrument than abuts the gap, unless
            # the gap is between data from two different instruments
            inst1 = spec[spec['w0'] > midpt]['instrument'][0]
            inst0 = spec[spec['w1'] < midpt]['instrument'][-1]
            if inst0 == inst1:
                span[spec['instrument'] != inst0] = False

            # try to filter out emission/absoprtion lines
            try:
                flags = specutils.split(wbins[span], flux[span], error[span],
                                        contcut=1.7, linecut=1.7, contfit=n,
                                        silent=silent)
                cont = (flags == 3)
                wb, f, e = [a[span][cont] for a in [wbins, flux, error]]
                if not silent:
                    pctmasked = 100.0 - 100.0 * np.sum(cont) / np.sum(span)
                    print ('succesfully masked lines including {:.1f}% of the '
                           'data'.format(pctmasked))
            except ValueError:
                if not silent:
                    print 'couldn\'t separate out continuum points'
                wb, f, e = [a[span] for a in [wbins, flux, error]]

            # fit polynomial to data
            poly = specutils.polyfit(wb, f, n, e)[2]
        else:
            wspan = [spec['w0'][fit_pts][0], spec['w1'][fit_pts][-1]]

        if resolution is None and findgaps:
            # compute average resolution in the span around gap
            dw = np.diff(wbins[span], 1)
            resolution = np.mean(dw)

        if findgaps:
            # make a grid to cover the gap
            m = round(width / resolution)
            gridedges = np.linspace(gr[0], gr[1], m + 1)
            gapbins = utils.edges2bins(gridedges)

        if not silent:
            print ('gap from {:.2f}-{:.2f} filled with order {} polynomial'
                   ''.format(gr[0], gr[1], n))

        # compute value of polynomial on the grid
        gapflux = poly(gapbins)[0]

        # add a comment about what will be filled to put in the spectbl meta
        comments.append('{}-{} filled with order {} polynomial fit to the '
                        'continuum within {}-{}'
                        ''.format(gr[0], gr[1], n, wspan[0], wspan[1]))

        # make a spectbl to fill in the gap
        w0, w1 = gapbins.T
        inst = rc.getinsti('mod_gap_fill-')
        star = spec.meta['STAR']
        name = spec.meta['NAME']
        gapspec = utils.vecs2spectbl(w0, w1, gapflux, instrument=inst,
                                     star=star, name=name)
        gapspecs.append(gapspec)

    # stack 'em
    if findgaps:
        # split at all gaps
        dataspecs = utils.gapsplit(spec)
    else:
        # if gapbins provided, just plit the spectrum at the specified gap
        blu = spec[spec['w1'] <= gapranges[0, 0]]
        red = spec[spec['w0'] >= gapranges[0, 1]]
        dataspecs = [blu, red]
    allspecs = [None] * (2 * len(gapspecs) + 1)
    allspecs[0::2] = dataspecs
    allspecs[1::2] = gapspecs
    filledspec = utils.vstack(allspecs, name=spec.meta['NAME'])
    filledspec.meta['COMMENT'].extend(comments)

    return filledspec


# FLARE STUFF
# ===========
def findflares(curveList, flagfactor=1.0, silent=True):
    """
    Return the start and stop time for all excursions (runs) from the median. Flag the excursions that have areas
    flagfactor greater than the mean. Takes a smoothed curve (t0, t1, rate, err), not a regular lightcurve.
    """

    ns = [len(curve[0]) for curve in curveList]
    expends = list(np.cumsum(ns))
    expstarts = [0] + expends[:-1]
    t0s, t1s, rates, errs = map(np.hstack, zip(*curveList))
    ts = (t0s + t1s) / 2.0
    clean = np.ones(len(rates), bool)
    count = 0
    while True:
        if count > 1000:
            break
        normrates = rates - np.median(rates[clean])
        runSliceList, areaList, begList, endList = [], [], [], []
        for i0, i1 in zip(expstarts, expends):
            t, t0, t1, normrate = [a[i0:i1] for a in [ts, t0s, t1s, normrates]]
            runslices = mnp.runslices(normrate)
            tslices = (t[runslices-1] + t[runslices]) / 2.0
            beg = np.insert(tslices, 0.0, t0[0])
            end = np.append(tslices, t1[-1])
            assert np.all((end - beg) > 0)
            begList.append(beg)
            endList.append(end)
            runSliceList.append(runslices + i0)

            # add slice between exposures
            runSliceList.append(i1)

            dt = np.append(np.diff(t0), t1[-1]-t1[-2])
            areas = normrate * dt
            areaList.append(mnp.splitsum(areas, runslices))

        runslices, areas, begs, ends = map(np.hstack, [runSliceList, areaList, begList, endList])
        runslices = np.insert(runslices, 0, 0)

        flare = areas > -(flagfactor * areas.min())
        oldclean = clean
        clean = np.ones(len(rates), bool)
        for i in np.nonzero(flare)[0]:
            i0, i1 = runslices[[i, i+1]]
            clean[i0:i1] = False

        if np.all(clean == oldclean):
            break

    # FIXME: this whole thing should probably go in flare stats, but in particular I need to add error on quiescent flux
    qrate = np.median(rates[clean])
    Fpeaks, FpeakErrs, tpeaks, ratios, ratioerrs = [], [], [], [], []
    for irun in range(len(areas)):
        i0, i1 = runslices[[irun, irun+1]]
        ipeak = np.argmax(np.abs(rates[i0:i1])) + i0
        Fpeaks.append(rates[ipeak] - qrate)
        FpeakErrs.append(errs[ipeak])
        ratios.append(rates[ipeak]/qrate)
        # FIXME: this should account for error in qrate
        ratioerrs.append(errs[ipeak]/qrate)
        tpeaks.append(ts[ipeak])

    return begs, ends, flare, Fpeaks, FpeakErrs, tpeaks, ratios, ratioerrs


def computeFlareStats(fitsphotons, begs, ends, flares, waveranges, dist):
    """
    Compute photometric equivalent widths and absolute energies for the regions defined by begs and ends (in s,
    referenced to MJD0 in fitsphotons) excluding those flagged as flares when computing the mean rate. dist in pc.
    """

    # keep weight and time info for only the photons we want
    photons = fitsphotons['events'].data
    keep = mnp.inranges(photons['wavelength'], waveranges)
    photons = photons[keep]

    # compute mean rate and flux
    cleanranges = np.array([begs[~flares], ends[~flares]])
    ttotal = np.sum(cleanranges[1] - cleanranges[0]) # s
    keep = mnp.inranges(photons['time'], cleanranges)
    mnrate = np.sum(photons[keep]['epsilon']) / ttotal # cnts s-1
    mnflux = np.sum(photons[keep]['epera']) / ttotal # erg s-1 cm-2

    # figure out where time bin edges fit in photons
    i0 = np.searchsorted(photons['time'], begs)
    i1 = np.searchsorted(photons['time'], ends)

    # for each bin, compute PEW
    epssum = np.insert(photons['epsilon'].cumsum(), 0, 0.0)
    assert epssum[-1] < 1e308 # otherwise I shouldn't use cumsum
    totals = epssum[i1] - epssum[i0]
    dts = ends - begs
    rates = totals / dts
    PEWs = (rates - mnrate) / mnrate * dts
    assert np.all(np.isfinite(PEWs))

    # compute ratio to cutoff PEW
    # FIXME: getting some negative PEW ratios, specifically for GJ832 SiIV
    PEWratios = np.abs(PEWs / PEWs.min())

    # multiply by mean luminosity to get absolute flare energy
    dist = dist * 3.08567758e18 # pc to cm
    mnlum = mnflux * 4 * np.pi * dist**2
    Es = PEWs * mnlum

    return PEWs, Es, PEWratios, mnrate, mnflux


def cumfreq(PEWs, exptime):
    """
    Compute cumulative flare frequency given PEWs of detected flares within total exposure time exptime [s].
    """
    isort = np.argsort(PEWs)[::-1]
    cfs = np.zeros(len(PEWs))
    cfs[isort] = (np.arange(len(PEWs)) + 1.0) / (exptime/3600.0/24.0)
    return cfs


def auto_flares(star, bands, inst, label, dt=1.0, silent=False):

    ph, photons = io.readphotons(star, inst)

    nexp = len(ph['gti'].data['obsids'])
    expt = ph[0].header['EXPTIME']

    curves = []
    groups = [range(len(bands))]
    for i in range(nexp):
        ii = (photons['expno'] == i)
        curve = sp.spectral_curves(photons['time'][ii], photons['wavelength'][ii], eps=photons['epsilon'][ii],
                                   tbins=dt, bands=bands, groups=groups)
        tedges, cps, err = zip(*curve)[0]
        t0, t1 = tedges[:-1], tedges[1:]
        curves.append([t0, t1, cps, err])

    begs, ends, flares, Fpeaks, FpeakErrs, tpeaks, ratios, ratioerrs = findflares(curves, silent=silent)

    dist = sc.quickval(db.proppath, star, 'dist')
    pews, Es, pewratios, mnrate, mnflux = computeFlareStats(ph, begs, ends, flares, bands, dist)

    assert pews.min() < 0

    cfs = cumfreq(pews, expt)

    data = [begs, ends, tpeaks, pews, flares, cfs, Es, pewratios, Fpeaks, FpeakErrs, ratios, ratioerrs]
    names = ['start', 'stop', 'peak', 'PEW', 'flare', 'cumfreq', 'energy', 'PEWratio', 'Fpk', 'Fpkerr', 'pkratio',
             'pkratioerr']
    units = ['s', 's', 's', 's', '', 'd-1', 'erg', '', 'counts s-1', 'counts s-1', '', '']
    descriptions = ['start time of excursion referenced to MJD0', 'end time of excursion referenced to MJD0',
                    'time of flare peak referenced to MJD0',
                    'photometric equivalent width', 'excursion flagged as a flare',
                    'cumulative frequency of flares of >= PEW', 'absolute energy radiated by flare in band',
                    'ratio of the PEW to the minimum PEW measured in the lightcurve',
                    'peak flux of flare', 'error in peak flux of flare', 'ratio of peak to quiescent flux',
                    'error in ratio of peak to quiescent flux']
    cols = []
    for d, n, u, dc in zip(data, names, units, descriptions):
        cols.append(Table.Column(d, n, unit=u, description=dc))
    flareTable = Table(cols)
    flareTable.meta['STAR'] = star
    flareTable.meta['EXPTIME'] = expt
    flareTable.meta['MJD0'] = ph[0].header['MJD0']
    flareTable.meta['QSCTCPS'] = mnrate
    flareTable.meta['QSCTFLUX'] = mnflux
    flareTable.meta['DT'] = dt

    for i, (w0, w1) in enumerate(bands):
        key0, key1 = 'bandbeg{}'.format(i), 'bandend{}'.format(i)
        flareTable.meta[key0] = w0
        flareTable.meta[key1] = w1

    flareTable.write(db.flarepath(star, inst, label), format='fits', overwrite=True)


def match_flares(star, inst, bandlabels='all', masterband='broad130a', flarecut=None):
    """
    Match the flares from the tables of flares for the bands sepcified by bandlables for specified star and instrument.
    Compute stats.
    """
    if bandlabels == 'all':
        fs = db.findfiles(rc.flaredir, inst, star, 'flares', fullpaths=True)
        bandlabels = [db.parse_info(f, 4, 5) for f in fs]
    try:
        bandlabels.remove(masterband)
    except ValueError:
        pass

    masterCat, _ = io.readFlareTbl(star, inst, masterband)

    colnames = masterCat.colnames
    masternames = [masterband + ' ' + name for name in colnames]
    matchCat = Table(data=masterCat, names=masternames)

    for band in bandlabels:
        for name in colnames:
            newname = band + ' ' + name
            matchCat[newname] = np.nan

    for band in bandlabels:
        bandCat, _ = io.readFlareTbl(star, inst, band)
        bandRanges = np.array([bandCat['start'], bandCat['stop']]).T
        for i, flare in enumerate(masterCat):
            if flarecut is None and not flare['flare']:
                continue
            if flarecut is not None and flare['PEWratio'] < flarecut:
                continue
            masterRange = np.array([[flare['start'], flare['stop']]])
            duration = flare['stop'] - flare['start']
            overlapping, _ = utils.argoverlap(bandRanges, masterRange, 'loose')
            slimCat = bandCat[overlapping & bandCat['flare']]
            overranges = zip(slimCat['start'], slimCat['stop'])
            overranges = mnp.range_intersect(overranges, [masterRange])
            overlap = overranges[:,1] - overranges[:,0]
            if np.any(overlap/duration) > 0.0:
                j = np.argmax(overlap)
                for name in bandCat.colnames:
                    matchCat[i][band + ' ' + name] = slimCat[j][name]

    return matchCat


def combine_flarecats(bandname, inst, flarecut=1.0, stars='all'):
    fs = db.findfiles(rc.flaredir, inst, bandname, 'flares', fullpaths=True)
    if stars != 'all':
        hasStar = lambda f: any([s in f for s in stars])
        fs = filter(hasStar, fs)

    tbls, dts, bands = [], [], []
    expt = 0.0
    for f in fs:
        star = db.parse_info(f, 3, 4)
        flareTable = Table.read(f, format='fits')
        expt += flareTable.meta['EXPTIME']

        # store start, stop, peak relative times under new names
        for key in ['start', 'peak', 'stop']:
            flareTable[key + ' rel'] = flareTable[key]

        # change begs and ends to mjd
        mjd0 = flareTable.meta['MJD0']
        flareTable['start'] = flareTable['start']/24.0/3600.0 + mjd0
        flareTable['stop'] = flareTable['stop']/24.0/3600.0 + mjd0
        flareTable['peak'] = flareTable['peak']/24.0/3600.0 + mjd0
        flareTable['start'].unit, flareTable['stop'].unit, flareTable['peak'].unit = 'mjd', 'mjd', 'mjd'

        # cull non-flares
        keep = flareTable['flare'] & (flareTable['PEWratio'] > flarecut)
        flareTable = flareTable[keep]

        # add some columns specific to table
        n = len(flareTable)
        bandname = db.parse_info(f, 4, 5)
        flareTable['inst'] = Table.Column([inst]*n, description='instrument', unit='', dtype=str)
        flareTable['star'] = Table.Column([star]*n, unit='', dtype=str)
        flareTable['bandname'] = Table.Column([bandname]*n, unit='', dtype=str)

        # record dt and bands to check that all are the same later
        dts.append(flareTable.meta['DT'])
        keys = flareTable.meta.keys()
        nbands = sum(map(lambda s: 'BANDBEG' in s, keys))
        band0 = [flareTable.meta['BANDBEG' + str(i)] for i in range(nbands)]
        band1 = [flareTable.meta['BANDEND' + str(i)] for i in range(nbands)]
        bands.append(np.array(zip(band0, band1)))

        # metadata will just cause merge conflicts
        for key in flareTable.meta:
            del flareTable.meta[key]
        del flareTable['flare']

        tbls.append(flareTable)

    assert np.allclose(dts, dts[0])
    assert np.allclose(bands[:-1], bands[1:])

    # compute cum. freq. of flare PEWs for all stars
    tbl = vstack(tbls)
    tbl.sort('PEW')
    tbl.reverse()
    tbl.meta['EXPTIME'] = expt
    tbl.meta['DT'] = dts[0]
    tbl.meta['BANDS'] = bands[0]
    tbl['cumfreqPEW'] = cumfreq(tbl['PEW'], expt)
    tbl['cumfreqPEW'].description = 'cumuluative frequency of all flares with >= PEW for the instrument'
    tbl['cumfreqE'] = cumfreq(tbl['energy'], expt)
    tbl['cumfreqE'].description = 'cumuluative frequency of all flares with >= energy for the instrument'
    tbl['cumfreqPEW'].unit = tbl['cumfreqE'].unit = 'd-1'

    return tbl


def auto_curve(star, inst, bands, dt, appx=True, groups=None, fluxed=False):
    ph, photons = io.readphotons(star, inst)

    # make bins
    gtis = ph['gti'].data
    tbinList = []
    for t0, t1 in zip(gtis['start'], gtis['stop']):
        if appx:
            n = round((t1 - t0) / dt)
            tbinList.append(np.linspace(t0, t1, n+1))
        else:
            tbinList.append(np.arange(t0, t1, dt))
    badbins = np.cumsum([len(b) for b in tbinList])[:-1] - 1
    tbins = np.hstack(tbinList)

    eps = 'epera' if fluxed else 'epsilon'
    p = photons
    tedges, rate, err = sp.spectral_curves(p['time'], p['wavelength'], eps=p[eps], tbins=tbins, bands=bands,
                                   groups=groups)
    t0, t1 = [], []
    for i in range(len(tedges)):
        rate[i], err[i] = [np.delete(a, badbins) for a in [rate[i], err[i]]]
        t0.append(np.delete(tedges[i][:-1], badbins))
        t1.append(np.delete(tedges[i][1:], badbins))

    assert all([np.all((tt1-tt0) < dt*2.0) for tt0,tt1 in zip(t0, t1)])
    return t0, t1, rate, err


def __inrange(spectbl, wr):
    in0, in1 = [mnp.inranges(spectbl[s], wr) for s in ['w0', 'w1']]
    return spectbl[in0 | in1]


def __edgesinrange(spectbl, wr):
    w = mnp.lace(spectbl['w0'], spectbl['w1'])
    duplicates = np.append((w[:-1] == w[1:]), False)
    w = w[~duplicates]
    return w[mnp.inranges(w, wr)]


def __same_instrument(spectbls):
    instruments = []
    for s in spectbls: instruments.extend(s['instrument'].data)
    instruments = np.array(instruments)
    if any(instruments[:-1] != instruments[1:]):
        raise ValueError('There are multiple instruments present in the '
                         'spectbls.')
    return instruments[0]


def __make_masks(welist, dqlist, dqmasks):
    # make master grid
    mwe = specutils.common_grid(welist)

    # rebin dq flags onto master grid, make masks, coadd those
    mwe_ins = [mnp.inranges(mwe, we[[0, -1]]) for we in welist]
    mwelist = [mwe[mwe_in] for mwe_in in mwe_ins]
    rdqs = map(mnp.rebin, mwelist, welist, dqlist, ['or'] * len(welist))
    masks = [(rdq & dqmask) > 0 for rdq, dqmask in zip(rdqs, dqmasks)]

    mmask = np.ones(len(mwe) - 1, bool)
    for mask, mwe_in in zip(masks, mwe_ins):
        i = np.nonzero(mwe_in)[0][:-1]
        mmask[i] = mmask[i] & mask

    # find the ranges where every spectrum is masked
    wbins = np.array([mwe[:-1], mwe[1:]]).T
    badranges = specutils.flags2ranges(wbins, mmask)

    # set each mask to false over those ranges
    masks = [(dq & dqmask) > 0 for dq, dqmask in zip(dqlist, dqmasks)]
    for we, mask in zip(welist, masks):
        inbad0, inbad1 = [mnp.inranges(w, badranges) for w in [we[:-1], we[1:]]]
        inbad = inbad0 | inbad1
        mask[inbad] = False

    return masks


def __same_star(spectbls):
    stars = np.array([s.meta['STAR'] for s in spectbls])
    if any(stars[1:] != stars[:-1]):
        raise ValueError('More than one target in the provided spectra.')
    return stars[0]
