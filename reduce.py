# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import numpy as np
from astropy.table import Table
from astropy.io import fits
import mypy.my_numpy as mnp
from math import sqrt, ceil, log10
from mypy import specutils
import database as db
import utils, io, settings
from spectralPhoton.hst.convenience import x2dspec
from itertools import combinations_with_replacement as combos
from scipy.stats import norm
from warnings import warn
from os.path import basename

colnames = settings.spectbl_format['colnames']
airglow_wavelengths = db.airglow_lines['wavelength'].data

def theworks(star, R=10000.0, dw=1.0, silent=False):

    if np.isnan(db.props['Teff'][star]):
        raise ValueError("Fool! You haven't entered Teff, etc. for {} yet."
                         "".format(star))

    try:
        settings.load(star)
    except IOError:
        warn("No settings file found for {}. Initializing one.".format(star))
        sets = settings.StarSettings(star)
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
    panspectrum(star, R=R, dw=dw) #panspec and Rspec

def panspectrum(star, R=10000.0, dw=1.0, savespecs=True, silent=False):
    """
    Coadd and splice the provided spectra into one panchromatic spectrum
    sampled at the native resolutions and constant R.

    Overlapping spectra will be normalized with the assumptions that they are
    listed in order of descending quality.
    """
    sets = settings.load(star)
    files, lyafile = db.panfiles(star)
    specs = io.read(files)

    # make sure all spectra are of the same star
    star = __same_star(specs)

    # make sure spectra are each from a single source
    for i,s in enumerate(specs):
        try:
            __same_instrument([s])
        except ValueError:
            raise ValueError('More than one instrument used in spectbl {}'.format(i))

    # carry out custom trims according to user-defined settings
    if not silent:
        print 'trimming spectra according to settings for {}'.format(star)
    for i in range(len(specs)):
        name = specs[i].meta['NAME']
        goodranges = sets.get_custom_range(name)
        if goodranges is not None:
            if not silent:
                print ('trimming spectra in {} to the ranges {}'
                       ''.format(name, goodranges))
            specs[i] = utils.keepranges(specs[i], goodranges)

    # remove airglow lines
    if not silent:
        print 'removing airglow from COS spectra'
    for i in range(len(specs)):
        spec = specs[i]
        name = spec.meta['NAME']
        if 'cos' in name:
            if 'g230l' in name:
                R_aperture = 165.0
            if 'g160m' in name or 'g130m' in name:
                R_aperture = 1500.0
            if not silent:
                print '\n\tremoving airglow lines from {}'.format(name)
            specrange = [spec['w0'][0], spec['w1'][-1]]
            relevant = mnp.inranges(airglow_wavelengths, specrange)
            rmv = lambda s, l: remove_line(s, l, 4, R_aperture, silent=silent)
            spec = reduce(rmv, airglow_wavelengths[relevant], spec)
            specs[i] = spec

    # normalize and splice according to input order
    spec = specs.pop(0)
    if not silent:
        print '\nstarting stitched spectrum with {}'.format(spec.meta['NAME'])
    while len(specs):
        addspec = specs.pop(0)

        if not silent:
            name = addspec.meta['NAME']
            specrange = [addspec['w0'][0], addspec['w1'][-1]]
            print ''
            print 'splicing in {}, covering {:.1f}-{:.1f}'.format(name, *specrange)

        overlap = utils.overlapping(spec, addspec)
        normit = not settings.dontnormalize(addspec)
        if not overlap and not silent:
            print '\tno overlap, so won\'t normalize'
        if not normit and not silent:
            print '\twon\'t normalize, cuz you said not to'
        if overlap and normit:
            if not silent:
                print '\tnormalizing within the overlap'
            addspec = normalize(spec, addspec, silent=silent)

        spec = smartsplice(spec, addspec, silent=silent)
    spec.meta['NAME'] = db.parse_name(db.panpath(star))

    #replace lya portion with model
    if lyafile is None:
        lyafile = db.lyafile(star)
    if not silent:
        print ('replacing section {:.1f}-{:.1f} with data from {lf}'
               ''.format(*settings.lyacut, lf=lyafile))
    lyaspec = io.read(lyafile)[0]
    lyaspec = normalize(spec, lyaspec, silent=silent)
    lyaspec = utils.keepranges(lyaspec, settings.lyacut)
    spec = splice(spec, lyaspec)
#    spec = cullrange(spec, settings.lyacut)
#    spec = smartsplice(spec, lyaspec)

    order, span = 4, 10
    if not silent:
        print ('filling in any gaps with an order {} polynomial fit to an '
               'area {}x the gap width'.format(order, span))
    spec = fill_gaps(spec, fill_with=order, fit_span=span, silent=silent)

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

    return spec,Rspec,dspec

def normalize(spectbla, spectblb, flagmask=False, silent=False):
    """
    Normalize the spectrum b to spectrum a.

    The spectra are assumed to be orded by quality. Thus, spectra[0] is the
    gold standard against which all others are normalized.

    spectblb is not normalized if:
    - spectbla has all 0.0 errors (meaning it's a model)
    - the normalization factor is consistent with 1.0 to greater than 95%
        confidence

    Errors in spectblb are augmented using the quadratic approximation
    according to the error in the normalization factor.
    """

    if not silent:
        names = [s.meta['NAME'] for s in [spectbla, spectblb]]

    #parse out the overlap
    overa, overb = utils.argoverlap(spectbla, spectblb, 'tight')

    #if speca has all zero errors (it's a model), don't normalize to it
    if np.all(spectbla[overa]['error'] == 0.0):
        if not silent:
            print ('the master spectrum {} has all zero errors, so {} will '
                   'not be normalized to it'.format(*names))
        return spectblb

    #rebin to the coarser spectrum
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

    #mask data with flags
    mask = np.zeros(len(ospeca), bool)
    if flagmask:
        flagged = (ospeca['flags'] > 0) | (ospecb['flags'] > 0)
        mask[flagged] = True
        if not silent:
            percent_flagged = np.sum(flagged) / float(len(ospeca)) * 100.0
            print ('{:.2f}% of the data that was flagged in one spectra or '
                   'the other. masking it out.'.format(percent_flagged))

    #mask data where speca has 0.0 errors
    zeroerr = (ospeca['error'] == 0.0)
    mask[zeroerr] = True
    if not silent:
        percent_zeroerr = np.sum(zeroerr) / float(len(ospeca)) * 100.0
        print ('{:.2f}% of the data in the master spectrum had zero error. '
               'masking it out'.format(percent_zeroerr))

    good = ~mask

    #compute normalization factor
    ospecs = [ospeca, ospecb]
    dw = wbins[:,1] - wbins[:,0]
    def getarea(spec):
        area = np.sum(spec['flux'][good]*dw[good])
        error = mnp.quadsum(spec['error'][good]*dw[good])
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
    if p > 0.05:
        if not silent:
            print 'secondary will not be normalized to master'
        return spectblb
    normfac = areas[0]/areas[1]
    normfacerr = sqrt((errors[0]/areas[1])**2 +
                      (areas[0]*errors[1]/areas[1]**2)**2)
    if not silent:
        print ('secondary will be normalized by a factor of {} ({})'
               ''.format(normfac, normfacerr))

    normspec = Table(spectblb, copy=True)
    nze = (normspec['error'] != 0.0)
    normspec['error'][nze] = mnp.quadsum([normspec['error'][nze]*normfac,
                                          normspec['flux'][nze]*normfacerr], axis=0)
    normspec['flux'] *= normfac
    normspec['normfac'] = normfac
    return normspec

def smartsplice(spectbla, spectblb, minsplice=0.05, silent=False):
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
    #sort the two spectra
    both = [spectbla, spectblb]
    key = lambda s: s['w0'][0]
    both.sort(key=key)
    spec0, spec1 = both
    if not silent:
        names = [s.meta['NAME'] for s in both]

    if not utils.overlapping(*both): #they don't overlap
        return utils.vstack(both)

    #get their overlap and the range of the overlap
    over0, over1 = utils.argoverlap(*both, method='loose')
    ospec0, ospec1 = spec0[over0], spec1[over1]
    oboth = [ospec0, ospec1]
    wr0, wr1 = [[s['w0'][0], s['w1'][-1]] for s in oboth]
    wr = [max(wr0[0], wr1[0]), min(wr0[1], wr1[1])]

    #if the spectra have gaps within the overlap, split them at their gaps,
    #sort them, and splice in pairs
    gaps0, gaps1 = map(utils.gapranges, both)
    gapsin0 = np.any(mnp.inranges(gaps0.flatten(), wr1))
    gapsin1 = np.any(mnp.inranges(gaps1.flatten(), wr0))
    if gapsin0 or gapsin1:
        specs = sum(map(utils.gapsplit, both), [])
        specs.sort(key=key)
        return reduce(smartsplice, specs)

    #if either spectrum has zeros for errors, don't use it for any of the
    #overlap
    allzeroerrs = lambda spec: np.all(spec['error'] == 0.0)

    #somehow the error for one of the phx entries is being changed to 0
    #from nan, so doing allnan on the original spectrum is a workaround
    ismodel0 = allzeroerrs(ospec0) or allzeroerrs(spec0)
    ismodel1 = allzeroerrs(ospec1) or allzeroerrs(spec1)
    if ismodel1 and not ismodel0:
        return splice(spec1, spec0)
    if ismodel0 and not ismodel1:
        return splice(spec0, spec1)
    if ismodel0 and ismodel1:
        return NotImplementedError('Not sure how to splice two models together.')

    #otherwise, find the best splice locations
    #get all edges within the overlap
    we0, we1 = [__edgesinrange(s, wr) for s in both]
    we = np.hstack([wr[0], we0, we1])
    we = np.unique(we) #also sorts the array
    wbins = utils.edges2bins(we)
    #wr[1] is already included because of how searchsorted works

    #rebin spectral overlap to we
    oboth = [rebin(o, wbins) for o in oboth]
    dw = np.diff(we)

    #get flux and variance and mask values with dq flags and nan values
    masks = [(spec['flags'] > 0) for spec in oboth]
    flus = [spec['flux']*dw for spec in oboth]
    sig2s = [(spec['error']*dw)**2 for spec in oboth]
    def maskitfillit(x, mask):
        x[mask] = 0.0
        x[np.isnan(x)] = 0.0
        return x
    flus = map(maskitfillit, flus, masks)
    sig2s = map(maskitfillit, sig2s, masks)

    sumstuff = lambda x: np.insert(np.cumsum(x), 0, 0.0)
    cf0, cf1 = map(sumstuff, flus)
    cv0, cv1 = map(sumstuff, sig2s)
    #this way, any portions where all variances are zero will result in a
    #total on nan for the signal to noise

    enclosed = (wr1[1] < wr0[1])
    if enclosed:
        wmid = (wr[1] + wr[0])/2.0
        mindw = minsplice*wmid
        if len(we) < 500:
            indices = np.arange(len(we))
        else:
            indices = np.round(np.linspace(0, len(we)-1, 500)).astype(int)
        ijs = combos(indices, 2)
        i, j = np.array([ij for ij in ijs]).T
        keep = (we[j] - we[i]) > mindw
        i, j = i[keep], j[keep]
        i, j = map(np.append, [i, j], [0, 0])

#        signal = cf0[i] + (cf1[j] - cf1[i]) + (cf0[-1] - cf0[j])
        var = cv0[i] + (cv1[j] - cv1[i]) + (cv0[-1] - cv0[j])
#        SN = signal/np.sqrt(var)

        #pick the best and splice the spectra
#        best = np.nanargmax(SN)
        best = np.nanargmin(var)
        i, j = i[best], j[best]
        cut0, cut1 = we[i], we[j]
        if i == j:
            return spec0
        if cut0 in we0:
            left = spec0[spec0['w1'] <= cut0]
            spec = splice(spec1, left)
        else:
            right = spec1[spec1['w0'] >= cut0]
            spec = splice(spec0, right)
        if cut1 in we0:
            right = spec0[spec0['w0'] >= cut1]
            spec = splice(spec, right)
        else:
            left = spec[spec['w1'] <= cut1]
            spec = splice(spec0, left)
        if not silent:
            print ('spectrum {} spliced into {} from {:.2f} to {:.2f}'
                   ''.format(names[1], names[0], cut0, cut1))

    #do the same, if not enclosed
    else:
        i = range(len(we))
#        signal = cf0[i] + (cf1[-1] - cf1[i])
        var = cv0[i] + (cv1[-1] - cv1[i])
#        SN = signal/np.sqrt(var)

#        best = np.nanargmax(SN)
        best = np.nanargmin(var)
        i = i[best]
        cut = we[best]
        if cut in we0:
            left = spec0[spec0['w1'] <= cut]
            spec = splice(spec1, left)
        else:
            right = spec1[spec1['w0'] >= cut]
            spec = splice(spec0, right)
        if not silent:
            print ('spectrum {} spliced into {} from {:.2f} onward'
                   ''.format(names[1], names[0], cut))

    return spec

def splice(spectbla, spectblb):
    """
    Replace spectrum a with spectrum b where they overlap.

    The outer bins of spectrum b are preserved, whereas the bins adjacent
    to the edges of spectrum b in spectrum a may be cut off. If so, the errors
    for the fractional bins are appropriately augmented assuming Poisson
    statistics and a constant flux within the original bins.
    """
    #if spectrum b has gaps, divide it up and add the pieces it into spectbla
    #separately
    if utils.hasgaps(spectblb):
        bspecs = utils.gapsplit(spectblb)
        return reduce(splice, bspecs, spectbla)

    #if the spectra do not overlap, just stack them
    if spectbla['w0'][0] >= spectblb['w1'][-1]:
        return utils.vstack([spectblb, spectbla])
    if spectblb['w0'][0] >= spectbla['w1'][-1]:
        return utils.vstack([spectbla, spectblb])

    #edges of the b spectrum
    wrb0, wrb1 = spectblb['w0'][0], spectblb['w1'][-1]
    w0a, w1a = spectbla['w0'], spectbla['w1']
    speclist = []

    #fit the left edge in first
    i = np.searchsorted(w0a, wrb0) #give the index of w0a just right of wrb0
    if i == 0: #the left edge of b is left of a altogether
        pass #spectrum does not star with any portion of a
    elif w1a[i-1] < wrb0: #the left edge of b is in a gap in a
        speclist.append(spectbla[:i])
    else: #the left edge of b is in bin i-1 of spec a
        #so we need to trim spectbla
        leftspec = Table(spectbla[:i], copy=True)
        dwold = w1a[i-1] - w0a[i-1]
        dwnew = wrb0 - w0a[i-1]
        leftspec['error'][-1] *= sqrt(dwold/dwnew)
        leftspec['w1'][-1] = wrb0
        speclist.append(leftspec)

    #now all of b goes in
    speclist.append(spectblb)

    #now the right edge
    j = np.searchsorted(w1a, wrb1, side='right')
    if j == len(w1a): #right side of b is beyond a
        pass #then a is totally within b, so only b will be returned
    elif w0a[j] > wrb1: #the right edge of b is a gap in a
        speclist.append(spectbla[j:])
    else: #the right edge of b is in bin j of spec a
        rightspec = Table(spectbla[j:], copy=True)
        dwold = w1a[j] - w0a[j]
        dwnew = w1a[j] - wrb1
        rightspec['error'][0] *= sqrt(dwold/dwnew)
        rightspec['w0'][0] = wrb1
        speclist.append(rightspec)

    spec = utils.vstack(speclist)

    metas = [s.meta for s in [spectbla, spectblb]]
    spec.meta['SOURCESPECS'] = np.hstack([m['SOURCESPECS'] for m in metas])
    spec.meta['FILENAME'] = ''
    spec.meta['NAME'] = 'stitched spectrum'
    return spec

def cullrange(spectbl, wrange):
    in0, in1 = [mnp.inranges(spectbl[s], wrange) for s in ['w0', 'w1']]
    keep = in0 & in1
    return spectbl[~keep]

def powerbin(spectbl, R=1000.0, lo=1.0, hi=None):
    """
    Rebin a spectrum onto a grid with constant resolving power.

    If the constant R grid cannot does not permit an integer number of bins
    within the original wavelength range, the remainder will be discarded.
    """
    start = spectbl['w0'][0]
    if start < lo: start = lo
    end = spectbl['w1'][-1] if hi is None else hi
    fac = (2.0*R + 1.0)/(2.0*R - 1.0)
    maxpow = ceil(log10(end/start)/log10(fac))
    powers = np.arange(maxpow)
    we = start*fac**powers
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
    star = __same_star(spectbls)

    sourcefiles = [s.meta['FILENAME'] for s in spectbls]

    listify = lambda s: [spec[s].data for spec in spectbls]
    w0, w1, f, e, expt, dq, inst, normfac, start, end = map(listify, colnames)
    we = [np.append(ww0,ww1[-1]) for ww0,ww1 in zip(w0,w1)]

    if any([np.any(n != 1.0) for n in normfac]):
        raise NotImplementedError("Can't deal with normfacs != 1.0 in coaddition.")

    weights = [1.0/ee**2 for ee in e] if weights == 'error' else expt
    if maskbaddata:
        dqmasks = map(settings.seriousdqs, sourcefiles)
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
    data = [v[goodbins] for v in [cw0,cw1,cf,ce,cexpt,dq,cinst,cnorm,cstart,cend]]
    cfile = db.coaddpath(sourcefiles[0])
    cname = db.parse_name(cfile)
    sourcespecs = [s.meta['NAME'] for s in spectbls]
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
        groups = [db.sourcespecfiles(star, config) for config in configs]

    for group in groups:
        if len(group) == 1 and not utils.isechelle(group[0]):
            if not silent:
                print 'single file for {}, moving on'.format(basename(group[0]))
            continue
        if not silent:
            names = map(basename, group)
            print 'coadding the files \n\t{}'.format('\n\t'.join(names))
        echelles = map(utils.isechelle, group)
        if any(echelles):
            weights = 'error'
            if not silent:
                print 'all files are echelles, so weighting by 1/error**2'
        else:
            weights = 'exptime'
            if not silent:
                print 'weighting by exposure time'
        spectbls = sum(map(io.read, group), [])
        if len(spectbls) > 1:
            coadd(spectbls, savefits=True, weights=weights, silent=silent)

def phxspec(Teff, logg=4.5, FeH=0.0, aM=0.0, repo=db.phxrepo):
    """
    Quad-linearly interpolates the available phoenix spectra to the provided
    values for temperature, surface gravity, metallicity, and alpha metal
    content.
    """
    grids = [db.phxTgrid, db.phxggrid, db.phxZgrid, db.phxagrid]
    pt = [Teff, logg, FeH, aM]

    #make a function to retrieve spectrum given grid indices
    def getspec(*indices):
        args = [grid[i] for grid,i in zip(grids, indices)]
        return io.phxdata(*args, repo=repo)

    #interpolate
    spec = mnp.sliminterpN(pt, grids, getspec)

    #make spectbl
    N = len(spec)
    err = np.zeros(N)
    expt,flags = np.zeros(N), np.zeros(N, 'i1')
    insti = settings.getinsti('mod_phx_-----')
    source = insti*np.ones(N,'i1')
    normfac, start, end = 1.0, 0.0, 0.0
    data = [db.phxwave[:-1], db.phxwave[1:], spec, err, expt, flags, source,
            normfac, start, end]
    return utils.list2spectbl(data)

def auto_phxspec(star, silent=False):
    Teff, kwds = db.phxinput(star)
    if not silent:
        print 'interpolating phoenix spectrum for {} with values'.format(star)
        kwds2 = dict(Teff=Teff, **kwds)
        print kwds2
    spec = phxspec(Teff, **kwds)
    spec.meta['STAR'] = star
    path = db.phxpath(star)
    spec.meta['NAME'] = db.parse_name(path)
    if not silent:
        print 'writing spectrum to {}'.format(path)
    io.writefits(spec, path, overwrite=True)

def auto_customspec(star, specfiles=None, silent=False):
    if specfiles is None:
        specfiles = db.allspecfiles(star)
    ss = settings.load(star)
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
            x1dfile = db.sourcespecfiles(star, config)
            if len(x1dfile) > 1:
                raise NotImplementedError('')
            else:
                x1dfile = x1dfile[0]
            x2dfile = x1dfile.replace('x1d','x2d')
            if not silent:
                print 'using x2dfile {}'.format(x2dfile)
            specfile = x1dfile.replace('x1d', 'custom_spec')
            dqmask = settings.seriousdqs(specfile)
            spec = x2dspec(x2dfile, x1dfile=x1dfile, bkmask=dqmask,
                           **custom['kwds'])

            #trim any nans
            isnan = np.isnan(spec['flux'])
            spec = spec[~isnan]

            #conform to spectbl standard
            datalist = [spec[s] for s in ['w0', 'w1', 'flux', 'error']]
            hdr = fits.getheader(x2dfile, extname='sci')
            expt, start, end = [hdr[s] for s in ['exptime', 'expstart', 'expend']]
            inst = db.getinsti(specfile)
            norm = 1.0
            datalist.extend([expt, spec['dq'], inst, norm, start, end])

            spectbl = utils.list2spectbl(datalist, star, specfile, '', [x2dfile])
            if not silent:
                 print 'saving custom extraction to {}'.format(specfile)
            io.writefits(spectbl, specfile, overwrite=True)
        else:
            raise NotImplementedError("No custom extractions defined for {}"
            "".format(config))

def rebin(spec, newbins):
    """Rebin the spectrum, dealing with gaps in newbins if appropriate. An
    exception is thrown if some newbins fall outside of spec."""

    #split newbins up at gaps
    w0, w1 = newbins.T
    if ~np.allclose(w0[1:], w1[:-1]): #gaps in the new edges
        splits = np.nonzero(~np.isclose(w0[1:], w1[:-1]))[0] + 1
        splitbins = np.split(newbins, splits, 0)
        specs = [rebin(spec, nb) for nb in splitbins]
        return utils.vstack(specs)

    #check for gaps in spec
    in0, in1 = [mnp.inranges(w, newbins) for w in utils.wbins(spec).T]
    spec = spec[in0 | in1]
    if utils.hasgaps(spec):
        warn('Spectrum has gaps within newbins.')
        #now remove newbin bins that fall outside of spectbl bins
        in0, in1 = [mnp.inranges(w, utils.wbins(spec)) for w in newbins.T]
        keep = in0 & in1
        return rebin(spec, newbins[keep])

    #rebin
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

    #spectbl accoutrments
    star, name, fn, sf = [spec.meta[s] for s in
                          ['STAR','NAME', 'FILENAME', 'SOURCESPECS']]

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
        radius =  w / minclip / 2.0
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
                                linecut=pline, contfit=n)

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
              gapbins=None, silent=False):
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
    if gapbins is None and not utils.hasgaps(spec):
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
    if gapbins is None:
        gapranges = utils.gapranges(spec)
    else:
        gapranges = np.array([[gapbins[0, 0], gapbins[-1, 1]]])

    gapspecs = []
    comments = []
    for gr in gapranges:
        width = gr[1] - gr[0]
        if fit_pts is None:
            # confine spectrum to just fit_span around gap
            midpt = (gr[0] + gr[1]) / 2.0
            radius = (fit_span * width) / 2.0
            wspan = [midpt - radius, midpt + radius]
            span = utils.argrange(spec, wspan)

            # try to filter out emission/absoprtion lines
            try:
                flags = specutils.split(wbins[span], flux[span], error[span],
                                        contfit=n)
                wb, f, e = [a[span][flags == 2] for a in [wbins, flux, error]]
            except ValueError:
                wb, f, e = [a[span] for a in [wbins, flux, error]]

            # fit polynomial to data
            poly = specutils.polyfit(wb, f, n, e)[2]
        else:
            wspan = [spec['w0'][fit_pts][0], spec['w1'][fit_pts][-1]]

        if resolution is None and gapbins is None:
            # compute average resolution in the span around gap
            dw = np.diff(wbins[span], 1)
            resolution = np.mean(dw)

        if gapbins is None:
            # make a grid to cover the gap
            m = round(width / resolution)
            gridedges = np.linspace(gr[0], gr[1], m + 1)
            gapbins = utils.bins2edges(gridedges)

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
        inst = settings.getinsti('mod_gap_fill-')
        star = spec.meta['STAR']
        name = spec.meta['NAME']
        gapspec = utils.vecs2spectbl(w0, w1, gapflux, instrument=inst,
                                     star=star, name=name)
        gapspecs.append(gapspec)

    # stack 'em
    if gapbins is None:
        # split at all gaps
        dataspecs = utils.gapsplit(spec)
    else:
        # if gapbins provided, just plit the spectrum at the specified gap
        blu = spec[spec['w1'] <= gapranges[0,0]]
        red = spec[spec['w0'] >= gapranges[0,1]]
        dataspecs = [blu, red]
    allspecs = [None] * (2 * len(gapspecs) + 1)
    allspecs[0::2] = dataspecs
    allspecs[1::2] = gapspecs
    filledspec = utils.vstack(allspecs, name=spec.meta['NAME'])
    filledspec.meta['COMMENT'].extend(comments)

    return filledspec

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
    #make master grid
    mwe = specutils.common_grid(welist)

    #rebin dq flags onto master grid, make masks, coadd those
    mwe_ins = [mnp.inranges(mwe, we[[0,-1]]) for we in welist]
    mwelist = [mwe[mwe_in] for mwe_in in mwe_ins]
    rdqs = map(mnp.rebin, mwelist, welist, dqlist, ['or']*len(welist))
    masks = [(rdq & dqmask) > 0 for rdq, dqmask in zip(rdqs, dqmasks)]

    mmask = np.ones(len(mwe) - 1, bool)
    for mask, mwe_in in zip(masks, mwe_ins):
        i = np.nonzero(mwe_in)[0][:-1]
        mmask[i] = mmask[i] & mask

    #find the ranges where every spectrum is masked
    wbins = np.array([mwe[:-1], mwe[1:]]).T
    badranges = specutils.flags2ranges(wbins, mmask)

    #set each mask to false over those ranges
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