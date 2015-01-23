# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import numpy as np
from astropy.table import Table, Column, vstack
from astropy.io import fits
import mypy.my_numpy as mnp
from math import sqrt, floor, log10
from mypy import specutils
import database as db
import utils, io, settings
from spectralPhoton.hst.convenience import x2dspec
from itertools import combinations

def panspectrum(star, R=1000.0, savespecs=True):
    """
    Coadd and splice the provided spectra into one panchromatic spectrum
    sampled at the native resolutions and constant R.
    
    Overlapping spectra will be normalized with the assumptions that they are 
    listed in order of descending quality. 
    """
    files = db.panfiles(star)
    specs = sum(map(io.read, files), [])
    
    #make sure all spectra are of the same star
    star = __same_star(specs)
    
    #make sure spectra are each from a single source
    for i,s in enumerate(specs):
        try: 
            __same_instrument([s])
        except ValueError:
            raise ValueError('More than one instrument used in spectbl {}'.format(i))
    
    #parse the modeled from the observed spectra
#    ismodel = lambda spec: 'mod' in spec.meta['FILENAME']
#    modelspecs = filter(ismodel, specs)
#    obsspecs = filter(lambda spec: not ismodel(spec), specs)
    
    #clip out any lya to leave that space for the reconstruction
    specs = [cullrange(s, settings.lyacut) for s in specs]
    
    #normalize and splice according to input order
    spec = specs.pop(0)
    while len(specs):
        addspec = specs.pop(0)
        if __overlapping(spec, addspec) and ~settings.dontnormalize(addspec):
            addspec = normalize(spec, addspec)
        spec = smartsplice(spec, addspec)
        
#    for i in range(N):
#        if settings.dontnormalize(specs[i]): continue
#        for j in np.arange(i+1,N):
#            if settings.dontnormalize(specs[j]): continue
#            if __overlapping(specs[i], specs[j]):
#                specs[j] = normalize(specs[i], specs[j])
#    
#    #splice together all the measured spectra based on S/N
#    #FIXME: for now, sort by wavelength because splicing over gaps doesn't work
#    #should probably generalize the splicing later, however
#    specs.sort(key = lambda s: s['w1'][-1])
#    catspec = reduce(smartsplice, specs)
    
    #splice the full extent of the models in
#    catspec = reduce(splice, modelspecs, catspec)
    
    #resample at constant R
    Rspec = powerbin(spec, R)
    
    if savespecs:
        #%% save to fits
        paths = [db.panpath(star), db.Rpanpath(star)]
        for spec, path in zip([spec, Rspec], paths):
            io.writefits(spec, path, overwrite=True)
            
    return spec,Rspec

def normalize(spectbla, spectblb, SNcut=2.0, method='area'):
    """
    Normalize the spectrum b to spectrum a. 
    
    The spectra are assumed to be orded by quality. Thus, spectra[0] is the
    gold standard against which all others are normalized. Use only points with
    S/N greater than SNcut when computing medians.
    """
    both = [spectbla, spectblb]
    
    #parse out the overlap
    over = __argoverlap(*both)
    ospecs = [s[o] for s,o in zip(both,over)]
    
    if len(over) == 0:
        return spectblb
    
    #make S/N cut
    SNs = [s['flux']/s['error'] for s in ospecs]
    fluxes = [s['flux'] for s in ospecs]
    for flux,SN in zip(fluxes,SNs): flux[SN < SNcut] = np.nan
    
    #compute normalization factor
    if method == 'meidan':
        meds = [np.nanmedian(flux) for flux in fluxes]
        normfac = meds[0]/meds[1]
    elif method == 'area':
        def getarea(spec):
            dw = spec['w1'] - spec['w0']
            a_elems = spec['flux']*dw
            return np.nansum(a_elems)
        oareas = map(getarea, ospecs)
        normfac = oareas[0]/oareas[1]
    
    normspec = Table(spectblb, copy=True)
    normspec['flux'] *= normfac
    normspec['error'] *= normfac
    return normspec

def smartsplice(spectbla, spectblb):
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
    """
    #sort the two spectra
    both = [spectbla, spectblb]
    both.sort(key = lambda s: s['w0'][0])
    
    #fill any gaps in the spectra
    #FIXME: don't use gapless spectra for splicing
    both = map(__fillgaps, both)
    
    #just for convenience
    spec0, spec1 = both
    
    #get their ranges
    wr0, wr1 = [[s['w0'][0], s['w1'][-1]] for s in both]
    
    #get overlapping wavelengths
    wr = [max(wr0[0], wr1[0]), min(wr0[1], wr1[1])]
    
    #splice according to different overlap situations
    if wr[0] >= wr[1]: #they don't overlap
        return utils.vstack(both)
    else: #they do overlap
        #get the loose (one edge beyound wr) overlap of each spectrum
        ospec0, ospec1 = [__inrange(s, wr) for s in both]
        
        #if either spectrum has nans for errors, don't use it for any of the
        #overlap
        def allnanerrs(ospec):
            nans = np.isnan(ospec['error'])
            return np.all(nans)
        if allnanerrs(spec0):
            return splice(spec0, spec1)
        if allnanerrs(both[1]):
            return splice(spec1, spec0)
            
        #otherwise, find the best splice locations
        #get all edges within the overlap
        we0, we1 = [__edgesinrange(s, wr) for s in both]
        we = np.hstack([we0, we1, wr])
        we = np.sort(we)
        
        #function to compute total signal/noise for different splice locations
        def fv(ospec, wr):
            dw = wr[1] - wr[0]
            wold = np.append(ospec['w0'], ospec['w1'][-1])
            flux, err = specutils.rebin(wr, wold, ospec['flux'], ospec['error'])
            f = flux*dw
            v = (err*dw)**2
            return f, v
        
        SN = []
        enclosed = (wr1[1] < wr0[1])
        
        if enclosed:
            #get all possible combinations of cuts
            cuts = combinations(we, 2)
            cuts = filter(lambda c: c[1] >= c[0], cuts)
            
            #computer overall SN for each
            for cut in cuts:
                ranges = [[wr[0], cut[0]], cut, [cut[1], wr[1]]]
                left, mid, right = map(fv, [ospec0, ospec1, ospec0], ranges)
                fs, vs = zip(left, mid, right)
                SN.append(sum(fs)/sqrt(sum(vs)))
            
            #pick the best and splice the spectra
            best = np.argmax(SN)
            cut = cuts[best]
            if cut[0] in we0:
                left = spec0[spec0['w1'] <= cut[0]]
                spec = splice(spec1, left)
            else:
                right = spec1[spec1['w0'] >= cut[0]]
                spec = splice(spec0, right)
            if cut[1] in we0:
                right = spec0[spec0['w0'] >= cut[1]]
                spec = splice(spec, right)
            else:
                left = spec[spec['w1'] <= cut[1]]
                spec = splice(spec, left)
            
        #do the same, if not enclosed
        else:
            cuts = we[:-1]
            for cut in cuts:
                ranges = [[wr[0], cut], [cut, wr[1]]]
                left, right = map(fv, [ospec0, ospec1], ranges)
                fs, vs, = zip(left, right)
                SN.append(sum(fs)/sqrt(sum(vs)))
                
        best = np.argmax(SN)
        cut = cuts[best]
        if cut in we0:
            left = spec0[spec0['w1'] <= cut]
            spec = splice(spec1, left)
        else:
            right = spec1[spec1['w0'] >= cut]
            spec = splice(spec0, right)
    return spec

def splice(spectbla, spectblb):
    """
    Replace spectrum a with spectrum b where they overlap.
    
    The outer bins of spectrum b are preserved, whereas the bins adjacent
    to the edges of spectrum b in spectrum a may be cut off. If so, the errors
    for the fractional bins are appropriately augmented assuming Poisson
    statistics and a constant flux within the original bins.
    
    The spectra are assumed to be gapless.
    """
    Na = len(spectbla)
    
    #define a function to adjust a cut off spectral element (row in spectbl)
    def cutoff(specrow, wnew, clipside):
        dwold = specrow['w1'] - specrow['w0']
        if clipside == 'left':
            dwnew = specrow['w1'] - wnew
            specrow['w0'] = wnew
        else:
            dwnew = wnew - specrow['w0']
            specrow['w1'] = wnew
        specrow['error'] *= sqrt(dwold/dwnew)
        return specrow
    
    #determine where the endpts of spectblb fall in spectbla
    wedges_a = np.append(spectbla['w0'], spectbla['w1'][-1])
    wrange_b = [spectblb['w0'][0], spectblb['w1'][-1]]
    args = np.searchsorted(wedges_a, wrange_b) - 1
    
    #deal with overlap
    speclist = []
    if args[0] == Na: #the left side of b is right of a
        speclist.extend([spectbla, spectblb])
    elif args[0] >= 0: #the left side of b is in a
        leftspec = Table(spectbla[:args[0]+1], copy=True)
        leftspec[-1] = cutoff(leftspec[-1], spectblb[0]['w0'], 'right')
        speclist.extend([leftspec,spectblb])
    else: #the left side of b is left of a
        speclist.append(spectblb)
    if args[1] == -1: #if the right side of b is left of a
        speclist.append(spectbla)
    elif args[1] < Na: #if the right side of b is in a
        #TODO: check
        rightspec = Table(spectbla[args[1]:], copy=True)
        rightspec[0] = cutoff(rightspec[0], spectblb[-1]['w1'], 'left')
        speclist.append(rightspec)
    
    return utils.vstack(speclist)

def cullrange(spectbl, wrange):
    in0, in1 = [mnp.inranges(spectbl[s], wrange) for s in ['w0', 'w1']]
    cull = in0 & in1
    return spectbl[~cull]

def powerbin(spectbl, R=1000.0, lowlim=1.0):
    """
    Rebin a spectrum onto a grid with constant resolving power.
    
    If the constant R grid cannot does not permit an integer number of bins
    within the original wavelength range, the remainder will be discarded.
    """
    start = spectbl['w0'][0]
    if start < lowlim: start = lowlim
    end = spectbl['w1'][-1]
    maxpow = floor(log10(end/start)/log10((2.0*R + 1.0)/(2.0*R - 1.0)))
    powers = np.arange(maxpow)
    w = start**powers
    return rebin(spectbl, w)
        
def coadd(spectbls, maskbaddata=True, savefits=False):
    """Coadd spectra in spectbls."""
    inst = __same_instrument(spectbls)
    star = __same_star(spectbls)
    
    sourcefiles = [s.meta['FILENAME'] for s in spectbls]
    
    #FIXME: if all spectra are masked in the same place, don't mask any
    listify = lambda s: [spec[s].data for spec in spectbls]
    cols = ['w0','w1','flux','error','exptime','flags']
    w0, w1, f, e, expt, dq = map(listify, cols)
    we = [np.append(ww0,ww1[-1]) for ww0,ww1 in zip(w0,w1)]
    if maskbaddata:
        spectrograph = db.parse_spectrograph(sourcefiles[0])
        dqmask = settings.dqmask[spectrograph]
        masks = __make_masks(we, dq, dqmask)
        cwe, cf, ce, cexpt, dq = specutils.coadd(we, f, e, expt, dq, masks)
    else:
        cwe, cf, ce, cexpt, dq = specutils.coadd(we, f, e, expt, dq)
    
    cw0, cw1 = cwe[:-1], cwe[1:]
    goodbins = (cexpt > 0)
    cw0,cw1,cf,ce,cexpt,dq = [v[goodbins] for v in [cw0, cw1, cf, ce, cexpt, dq]] 
       
    spectbl = utils.vecs2spectbl(cw0,cw1,cf,ce,cexpt,dq,inst,star,None,
                                 sourcefiles)
    if savefits:
        cfile = db.coaddpath(sourcefiles[0])
        io.writefits(spectbl, cfile, overwrite=True)
        spectbl.meta['FILENAME'] = cfile
    return spectbl
    
def auto_coadd(star, configs=None):
    if configs is None:
        groups = db.specfilegroups(star)
    else:
        if type(configs) is str: configs = [configs]
        groups = [db.configfiles(star, config) for config in configs]
        
    for group in groups:
        spectbls = sum(map(io.read, group), [])
        coadd(spectbls, savefits=True)
    
def phxspec(Teff, logg=4.5, FeH=0.0, aM=0.0, repo=db.phxpath):
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
    err = np.ones(N)*np.nan
    expt,flags = np.zeros(N), np.zeros(N, 'i1')
    insti = settings.instruments.index('mod_phx_-----')
    source = insti*np.ones(N,'i1')
    data = [db.phxwave[:-1], db.phxwave[1:], spec, err, expt, flags, source]
    return utils.list2spectbl(data, '', '')

def auto_phxspec(star):
    Teff, kwds = db.phxinput(star)
    spec = phxspec(Teff, **kwds)
    spec.meta['STAR'] = star
    path = db.phxspecpath(star)
    io.writefits(spec, path, overwrite=True)
    return spec

def auto_customspec(star, specfiles=None):
    if specfiles is None:
        specfiles = db.allspecfiles(star)
    ss = settings.load(star) 
    for custom in ss.custom_extractions:
        config = custom['config']
        if 'hst' in config:
            x1dfile = filter(lambda f: config in f, specfiles)[0]
            x2dfile = x1dfile.replace('x1d','x2d')
            specfile = x1dfile.replace('x1d', 'custom_spec')
            spectbl = x2dspec(x2dfile, x1dfile=x1dfile, **custom['kwds'])
            
            #conform to spectbl standard
            spectbl.rename_column('dq', 'flags')
            spectbl.meta['STAR'] = star
            spectbl.meta['SOURCEFILES'] = [x2dfile]
            try:
                inst = db.getinsti(specfile)
            except ValueError:
                inst = -99
            n = len(spectbl)
            instcol = Column([inst]*n, 'instrument', 'i1')
            expt = fits.getval(x2dfile, 'exptime', extname='sci')
            exptcol = Column([expt]*n, 'exptime')
            spectbl.add_columns([instcol, exptcol])
            
            io.writefits(spectbl, specfile, overwrite=True)
        else:
            raise NotImplementedError("No custom extractions defined for {}"
            "".format(config))

def rebin(spec, newedges):
    newedges = np.asarray(newedges)
    oldedges = np.append(spec['w0'], spec['w1'][-1])
    flux, error, flags = specutils.rebin(newedges, oldedges, spec['flux'], 
                                         spec['error'], spec['flags'])
    star, fn, sf = [spec.meta[s] for s in ['STAR', 'FILENAME', 'SOURCEFILES']]
    w0, w1 = newedges[:-1], newedges[1:]
    inst = np.array([spec['instrument'][0]]*len(flux))
    dold, dnew = np.diff(oldedges), np.diff(newedges)
    expt = mnp.rebin(newedges, oldedges, spec['exptime']*dold)/dnew
    return utils.vecs2spectbl(w0, w1, flux, error, expt, flags, inst, star, fn, sf) 



def __argoverlap(spectbl0, spectbl1):
    """ Find the (boolean) indices of the overlap of spectbl0 within spectbl1
    and the reverse.
    """
    both = [spectbl0, spectbl1]
    grids = [np.append(s['w0'], s['w1'][-1]) for s in both]
    over0w1 = mnp.inranges(grids[0], grids[1][[0,-1]])
    over1w0 = mnp.inranges(grids[1], grids[0][[0,-1]])
    return [over0w1[:-1], over1w0[:-1]]

def __inrange(spectbl, wr):
    in0, in1 = [mnp.inranges(spectbl[s], wr) for s in ['w0', 'w1']]
    return spectbl[in0 | in1]

def __edgesinrange(spectbl, wr):
    w = mnp.lace(spectbl['w0'], spectbl['w1'])
    duplicates = np.append((w[:-1] == w[1:]), False)
    w = w[~duplicates]
    return w[mnp.inranges(w, wr)]

def __fillgaps(spectbl, fill_value=np.nan):
    """Fill any gaps in the wavelength range of spectbl with spome value."""
    w0, w1 = spectbl['w0'], spectbl['w1']
    gaps = ~np.isclose(w0[1:], w1[:-1])
    if ~np.any(gaps):
        return spectbl
    i = np.nonzero(gaps)[0]
    gw0 = w1[i]
    gw1 = w0[i+1]
    names = spectbl.colnames
    names.remove('w0')
    names.remove('w1')
    cols = [Column(gw0, 'w0'), Column(gw1, 'w1')]
    n = len(gw0)
    for name in names:
        cols.append(Column([fill_value]*n, name))
    gaptbl = Table(cols)
    filledtbl = vstack([spectbl, gaptbl])
    filledtbl.sort('w0')
    return filledtbl

def __overlapping(spectbl0, spectbl1):
    """Check if they overlap."""
    wr = lambda spec: [spec['w0'][0], spec['w1'][-1]]
    return any(np.digitize(wr(spectbl0), wr(spectbl1)) == 1)
    
def __same_instrument(spectbls):
    instruments = []
    for s in spectbls: instruments.extend(s['instrument'].data)
    instruments = np.array(instruments)
    if any(instruments[:-1] != instruments[1:]):
        raise ValueError('There are multiple instruments present in the '
                         'spectbls.')
    return instruments[0]

def __make_masks(welist, dqlist, dqmask):
    #make master grid
    mwe = specutils.common_grid(welist)
    
    #rebin dq flags onto master grid, make masks, coadd those
    mwe_ins = [mnp.inranges(mwe, we[[0,-1]]) for we in welist]
    mwelist = [mwe[mwe_in] for mwe_in in mwe_ins]
    rdqs = map(mnp.rebin_or, mwelist, welist, dqlist)
    masks = [(rdq & dqmask) > 0 for rdq in rdqs]
    
    mmask = np.ones(len(mwe) - 1, bool)
    for mask, mwe_in in zip(masks, mwe_ins):
        i = np.nonzero(mwe_in)[0][:-1]
        mmask[i] = mmask[i] & mask
    
    #find the ranges where every spectrum is masked
    wbins = np.array([mwe[:-1], mwe[1:]]).T
    badranges = specutils.flags2ranges(wbins, mmask)
    
    #set each mask to false over those ranges
    masks = [(dq & dqmask) > 0 for dq in dqlist]
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