# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
import mypy.my_numpy as mnp
from math import sqrt, ceil, log10
from mypy import specutils
import database as db
import utils, io, settings
from spectralPhoton.hst.convenience import x2dspec
from itertools import combinations_with_replacement as combos
from scipy.stats import norm

def panspectrum(star, R=10000.0, dR=1.0, savespecs=True):
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

    #clip out any lya to leave that space for the reconstruction
    for i,s in enumerate(specs):
        if 'mod_lya' not in s.meta['FILENAME']:
            specs[i] = cullrange(s, settings.lyacut)
    
    #normalize and splice according to input order
    spec = specs.pop(0)
    while len(specs):
        addspec = specs.pop(0)
        overlap = utils.overlapping(spec, addspec)
        normit = not settings.dontnormalize(addspec)
        if overlap and normit:
            addspec = normalize(spec, addspec)
        spec = smartsplice(spec, addspec)
    
    #resample at constant R and dR
    Rspec = powerbin(spec, R)
    newedges = np.arange(np.min(spec['w0']), np.max(spec['w1']), dR)
    newbins = utils.edges2bins(newedges)
    dspec = rebin(spec, newbins)
    
    if savespecs:
        #%% save to fits
        paths = [db.panpath(star), db.Rpanpath(star, R), db.dpanpath(star, dR)]
        for s, path in zip([spec, Rspec, dspec], paths):
            io.writefits(s, path, overwrite=True)
            
    return spec,Rspec,dspec

def normalize(spectbla, spectblb, flagmask=True):
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
    
    #parse out the overlap
    overa, overb = utils.argoverlap(spectbla, spectblb, 'tight')
    
    #if speca has all nan errors, don't normalize to it
    if np.all(spectbla[overa]['error'] == 0.0):
        return spectblb
    
    #rebin to the coarser spectrum
    if np.sum(overa) < np.sum(overb):
        ospeca = spectbla[overa]
        wbins = utils.wbins(ospeca)
        ospecb = rebin(spectblb, wbins)
    else:
        ospecb = spectblb[overb]
        wbins = utils.wbins(ospecb)
        ospeca = rebin(spectbla, wbins)
    
    #mask data with flags
    mask = np.zeros(len(ospeca))
    if flagmask:
        flagged = (ospeca['flags'] > 0) | (ospecb['flags'] > 0)
        mask[flagged] = True
        
    #mask data where speca has 0.0 errors
    zeroerr = (ospeca['error'] == 0.0)
    mask[zeroerr] = True
    good = ~mask
    
    #compute normalization factor
    ospecs = [ospeca, ospecb]
    dw = wbins[:,1] - wbins[:,0]
    def getarea(spec):
        area = np.sum(spec['flux'][good]*dw)
        error = mnp.quadsum(spec['error'][good]*dw)
        return area, error
    areas, errors = zip(*map(getarea, ospecs))
    diff = abs(areas[1] - areas[0])
    differr = mnp.quadsum(errors)
    p = 2.0 * (1.0 - norm.cdf(diff, loc=0.0, scale=differr))
    if p < 0.05:
        return spectblb
    normfac = areas[0]/areas[1]
    normfacerr = sqrt((errors[0]/areas[1])**2 + 
                      (areas[0]*errors[1]/areas[1]**2))
    
    normspec = Table(spectblb, copy=True)
    normspec['flux'] *= normfac
    normspec['error'] *= mnp.quadsum([normspec['error']*normfac,
                                      normspec['flux']*normfacerr], axis=0)
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
    key = lambda s: s['w0'][0]
    both.sort(key=key)
    spec0, spec1 = both
    
    if not utils.overlapping(*both): #they don't overlap
        return utils.vstack(both)
        
    #if the spectra have gaps within the overlap, split them at their gaps, 
    #sort them, and splice in pairs
    over0, over1 = utils.argoverlap(*both, method='loose')
    if utils.hasgaps(spec0[over0]) or utils.hasgaps(spec1[over1]):
        specs = sum(map(utils.gapsplit, both), [])
        specs.sort(key=key)
        return reduce(smartsplice, specs)
    
    #get their ranges
    wr0, wr1 = [[s['w0'][0], s['w1'][-1]] for s in both]
    
    #get overlapping wavelengths
    wr = [max(wr0[0], wr1[0]), min(wr0[1], wr1[1])]
    
    #splice according to different overlap situations
    #get the loose (one edge beyound wr) overlap of each spectrum
    ospec0, ospec1 = [__inrange(s, wr) for s in both]
    
    #if either spectrum has zeros for errors, don't use it for any of the
    #overlap
    allzeroerrs = lambda spec: np.all(spec['error'] == 0.0)
    
    #somehow the error for one of the phx entries is being changed to 0
    #from nan, so doing allnan on the original spectrum is a workaround
    ismodel0 = allzeroerrs(ospec0) or allzeroerrs(spec0)
    ismodel1 = allzeroerrs(ospec1) or allzeroerrs(spec1)
    if ismodel1 and not ismodel0:
        return splice(spec0, spec1)
    if ismodel0 and not ismodel1:
        return splice(spec1, spec0)
    if ismodel0 and ismodel1:
        return NotImplementedError('Not sure how to splice two models together.')
        
    #otherwise, find the best splice locations
    #get all edges within the overlap
    we0, we1 = [__edgesinrange(s, wr) for s in both]
    we = np.hstack([we0, we1, [wr[1]]])
    we = np.sort(we)
    wbins = utils.edges2bins(we)
    #wr[0] is already included because of how searchsorted works
    
    #rebin spectral overlap to we
    oboth = [ospec0, ospec1]
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
        if len(we) < 500:
            indices = np.arange(len(we))
        else:
            indices = np.round(np.linspace(0, len(we)-1, 500)).astype(int)
        ijs = combos(indices, 2)
        ijs = np.array(filter(lambda x: x[1] >= x[0], ijs))
        i, j = ijs.T
        
        signal = cf0[i] + (cf1[j] - cf1[i]) + (cf0[-1] - cf0[j])
        var = cv0[i] + (cv1[j] - cv1[i]) + (cv0[-1] - cv0[j])
        SN = signal/np.sqrt(var)
        
        #pick the best and splice the spectra
        best = np.nanargmax(SN)
        ij = ijs[best]
        cut = we[ij]
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
        i = range(len(we))
        signal = cf0[i] + (cf1[-1] - cf1[i]) 
        var = cv0[i] + (cv1[-1] - cv1[i])
        SN = signal/np.sqrt(var)
        
        best = np.nanargmax(SN)
        i = i[best]
        cut = we[best]
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
    j = np.searchsorted(w1a, wrb1)
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
    fac = (2.0*R + 1.0)/(2.0*R - 1.0)
    maxpow = ceil(log10(end/start)/log10(fac))
    powers = np.arange(maxpow)
    we = start*fac**powers
    wbins = utils.edges2bins(we)
    return rebin(spectbl, wbins)
        
def coadd(spectbls, maskbaddata=True, savefits=False):
    """Coadd spectra in spectbls."""
    inst = __same_instrument(spectbls)
    star = __same_star(spectbls)
    
    sourcefiles = [s.meta['FILENAME'] for s in spectbls]
    
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
    err = np.zeros(N)
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

def auto_customspec(star, specfiles=None):
    if specfiles is None:
        specfiles = db.allspecfiles(star)
    ss = settings.load(star) 
    for custom in ss.custom_extractions:
        config = custom['config']
        if 'hst' in config:
            x1dfile = db.sourcespecfiles(star, config)
            if len(x1dfile) > 1:
                raise NotImplementedError('')
            else:
                x1dfile = x1dfile[0]
            x2dfile = x1dfile.replace('x1d','x2d')
            specfile = x1dfile.replace('x1d', 'custom_spec')
            dqmask = settings.dqmask[db.parse_spectrograph(specfile)]
            spectbl = x2dspec(x2dfile, x1dfile=x1dfile, bkmask=dqmask,
                              **custom['kwds'])
            
            #conform to spectbl standard
            spectbl.rename_column('dq', 'flags')
            spectbl.meta['STAR'] = star
            spectbl.meta['SOURCEFILES'] = [x2dfile]
            inst = db.getinsti(specfile)
            n = len(spectbl)
            instcol = Column([inst]*n, 'instrument', 'i1')
            expt = fits.getval(x2dfile, 'exptime', extname='sci')
            exptcol = Column([expt]*n, 'exptime')
            spectbl.add_columns([instcol, exptcol])
            
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
        raise ValueError('Spectrum has gaps with newbins.')
        
    #rebin
    newedges = utils.bins2edges(newbins)
    oldedges = utils.wedges(spec)
    flux, error, flags = specutils.rebin(newedges, oldedges, spec['flux'], 
                                         spec['error'], spec['flags'])
                                         
    #spectbl accoutrments
    star, fn, sf = [spec.meta[s] for s in ['STAR', 'FILENAME', 'SOURCEFILES']]
    dold, dnew = np.diff(oldedges), np.diff(newedges)
    expt = mnp.rebin(newedges, oldedges, spec['exptime']*dold)/dnew
    inst = np.array([db.getinsti(fn)]*len(expt))
    
    return utils.vecs2spectbl(w0, w1, flux, error, expt, flags, inst, star, fn, sf) 

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