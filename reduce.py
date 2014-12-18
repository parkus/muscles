# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import numpy as np
from astropy.table import Table
import my_numpy as mnp
from math import sqrt, floor, log10
import specutils
import database as db
import utils, io

def panspectrum(spectbls, R=1000.0, savecoadds=True):
    """
    Coadd and splice the provided spectra into one panchromatic spectrum
    sampled at the native resolutions and constant R.
    
    Overlapping spectra will be normalized with the assumptions that they are 
    listed in order of descending quality. 
    """
    #make sure all spectra are of the same star
    __same_star(spectbls)
    
    #make sure spectra are each from a single source
    for i,s in enumerate(spectbls):
        try: 
            __same_instrument([s])
        except ValueError:
            raise ValueError('More than one instrument used in spectbl {}'.format(i))
    
    #coadd all spectra from the same configuration
    groups = db.group_by_instrument(spectbls)
    coadds = []
    for group in groups:
        filenames = [s.meta['filename'] for s in group]
        if len(group) == 1:
            coadds.extend(group)
        #FIXME: this is to avoid gaps, but i don't think it should be coded like this
        if len(set(filenames)) == 1:
            coadds.extend(group)
        else:
            cspec = coadd(group, savefits=savecoadds)
            coadds.append(cspec)
    
    #parse the modeled from the observed spectra
    ismodel = lambda spec: 'mod' in spec.meta['filename']
    modelspecs = filter(ismodel, coadds)
    obsspecs = filter(lambda spec: not ismodel(spec), coadds)
    
    #normalize the measured spectra according to their input order if they
    #overlap
    #FIXME: make sure this is working right
    N = len(obsspecs)
    for i in range(N):
        for j in np.arange(i+1,N):
            if __overlapping(obsspecs[i], obsspecs[j]):
                obsspecs[j] = normalize(obsspecs[i], obsspecs[j])
    
    #splice together all the measured spectra based on S/N
    #FIXME: for now, sort by wavelength because splicing over gaps doesn't work
    #should probably generalize the splicing later, however
    obsspecs.sort(key = lambda s: s['w1'][-1])
    catspec = reduce(smartsplice, obsspecs)
    
    #splice the full extent of the models in
    catspec = reduce(splice, modelspecs, catspec)
    
    #resample at constant R
    Rspec = powerbin(catspec, R)
    
    return catspec,Rspec

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
    
    #get their ranges
    wra, wrb = [[s['w0'][0], s['w1'][-1]] for s in both]
    
    #check for overlap
    if wrb[0] >= wra[1]: #they don't overlap
        return utils.vstack(both)
    else: #they overlap
        #resample to the finer spectrum to the resolution of the coarser
        #figure out the (boolean) indices of the overlap
        over = __argoverlap(*both)
        #count the number of bin edges in the overlap to quanitfy resolution
        res = map(np.sum, over)
        #determine which is the lower resolution to use as the base for comparison
        base = np.argmin(res)
        #confine spectra to just the overlap, rebinning the higher res spectrum
        i0,i1 = np.nonzero(over[base])[0][[0,-1]]
        basespec = both[base][i0+1:i1-1]
        w0base, w1base = basespec['w0'], basespec['w1']
        basegrid = np.append(w0base, w1base[-1])
        N = len(basegrid) - 1
        ospecs = [0,0]
        ospecs[base] = basespec
        ospecs[not base] = rebin(both[not base], basegrid) #rebin the finer spectrum
        
        #now figure out where to splice the two
        def bestsplice(spectbls): 
        #determine the cumulative signal to noise, summing bins for every
        #possible combo of the two spectra
            dw = np.diff(basegrid)
            fw = [s['flux']*dw for s in spectbls]
            vw = [(s['error']*dw)**2 for s in spectbls]
            cumsum = lambda x: np.append(0.0, np.cumsum(x))
            fsum0, vsum0 = map(cumsum, [fw[0], vw[0]])
            fsum1, vsum1 = map(cumsum, [fw[1][::-1], vw[1][::-1]])
            totf = fsum0 + fsum1[::-1]
            totv = vsum0 + vsum1[::-1]
            fullSN = totf/np.sqrt(totv)
            return np.argmax(fullSN)
            
        if wrb[1] > wra[1]: #just the ends of the spectra overlap
            #compute integrated SN for all possible splice positions, choose the best
            i = bestsplice(ospecs)
            globi = np.nonzero(over[1])[0][i]
            splicespec = both[1][globi:]
            return splice(both[0], splicespec)
        else: #both[1] is fully within both[0]
            #look for a single block of both[1] to put into both[0] that
            #maximizes the SN
            #start by finding left splice
            i0 = bestsplice(ospecs)
            temp = utils.vstack([ospecs[0][:i0], ospecs[1][i0:]])
            #now find the right splice
            i1 = bestsplice([temp, ospecs[1]])
            if i0 >= i1: return both[0]
            #TODO: these might be off by 1 since I shortened the base grid...
            globi0 = np.nonzero(over[1])[0][i0]
            globi1 = np.nonzero(over[1])[0][i1]
            splicespec = both[1][globi0:globi1]
            #TODO: deal with partial bins
            return splice(both[0], splicespec)

def splice(spectbla, spectblb):
    """
    Replace spectrum a with spectrum b where they overlap.
    
    The outer bins of spectrum b are preserved, whereas the bins adjacent
    to the edges of spectrum b in spectrum a may be cut off. If so, the errors
    for the fractional bins are appropriately augmented assuming Poisson
    statistics and a constant flux within the original bins.
    
    The spectra are assumed to be gapless.
    """
    Na = len(spectbla) + 1
    
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
    args = np.searchsorted(wedges_a, wrange_b)
    
    #deal with overlap
    speclist = []
    if args[0] == Na: #the left side of b is right of a
        speclist.extend([spectbla, spectblb])
    elif args[0] > 0: #the left side of b is in a
        leftspec = spectbla[:args[0]]
        leftspec[-1] = cutoff(leftspec[-1], spectblb[0]['w0'], 'right')
        speclist.extend([leftspec,spectblb])
    else: #the left side of b is left of a
        speclist.append(spectblb)
    if args[1] == 0: #if the right side of b is left of a
        speclist.append(spectbla)
    if args[1] < Na: #if the right side of b is in a
        #TODO: check
        rightspec = spectbla[args[1]-1:]
        rightspec[0] = cutoff(rightspec[0], spectblb[-1]['w1'], 'left')
        speclist.append(rightspec)
    
    return utils.vstack(speclist)

def powerbin(spectbl, R=1000.0, lowlim=1.0):
    """
    Rebin a spectrum onto a grid with constant resolving power.
    
    If the constant R grid cannot does not permit an integer number of bins
    within the original wavelength range, the remainder will be discarded.
    """
    start = spectbl['w0'][0]
    if start < lowlim: start = lowlim
    end = spectbl['w1'][-1]
    dwmin = start/R
    maxpow = floor(log10(end/start)/log10((2.0*R + 1.0)/(2.0*R - 1.0)))
    powers = np.arange(maxpow)
    w = start**powers
    return rebin(spectbl, w)
        
def coadd(spectbls, maskbaddata=True, savefits=False):
    """Coadd spectra in spectbls."""
    inst = __same_instrument(spectbls)
    star = __same_star(spectbls)
    
    sourcefiles = [s.meta['filename'] for s in spectbls]
    
    listify = lambda s: [spec[s].data for spec in spectbls]
    cols = ['w0','w1','flux','error','exptime','flags']
    w0, w1, f, e, expt, dq = map(listify, cols)
    we = [np.append(ww0,ww1[-1]) for ww0,ww1 in zip(w0,w1)]
    if maskbaddata:
        masks = [ddq > 0 for ddq in dq]
        cwe, cf, ce, cexpt = specutils.coadd(we, f, e, expt, masks)
        cw0, cw1 = cwe[:-1], cwe[1:]
        goodbins = (cexpt > 0)
        cw0,cw1,cf,ce,cexpt = [v[goodbins] for v in [cw0, cw1, cf, ce, cexpt]]
        dq = 0
    else:
        cwe, cf, ce, cexpt = specutils.coadd(we, f, e, expt)
        cw0, cw1 = cwe[:-1], cwe[1:]
        dq = np.nan
        
    spectbl = utils.vecs2spectbl(cw0,cw1,cf,ce,cexpt,dq,inst,star,None,
                                 sourcefiles)
    if savefits:
        cfile = db.coaddpath(sourcefiles[0])
        io.writefits(spectbl, cfile, overwrite=True)
        spectbl.meta['filename'] = cfile
    return spectbl
    
def phxspec(Teff, logg=4.5, FeH=0.0, aM=0.0, repo='.'):
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
        return io.readphx(*args, repo=repo)
    
    #interpolate
    spec = mnp.sliminterpN(pt, grids, getspec)
    
    #make spectbl
    N = len(spec)
    err = np.ones(N)*np.nan
    expt,flags = np.zeros(N), np.zeros(N, 'i1')
    insti = db.instruments.index('mod_phx_-----')
    source = insti*np.ones(N,'i1')
    data = [db.phxwave[:-1], db.phxwave[1:], spec, err, expt, flags, source]
    return utils.list2spectbl(data, '', '')

def rebin(spec, newedges):
    oldedges = np.append(spec['w0'], spec['w1'][-1])
    flux, error = specutils.rebin(newedges, oldedges, spec['flux'], spec['error'])
    star, fn, sf = [spec.meta[s] for s in ['star', 'filename', 'sourcefiles']]
    w0, w1 = newedges[:-1], newedges[1:]
    N = len(flux)
    inst = np.array([spec['instrument'][0]]*len(flux))
    flags = np.array([np.nan]*N)
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
    
def __overlapping(spectbl0, spectbl1):
    """Check if they overlap."""
    wr = lambda spec: [spec['w0'][0], spec['w1'][1]]
    return any(np.digitize(wr(spectbl0), wr(spectbl1)) == 1)
    
def __same_instrument(spectbls):
    instruments = []
    for s in spectbls: instruments.extend(s['instrument'].data)
    instruments = np.array(instruments)
    if any(instruments[:-1] != instruments[1:]):
        raise ValueError('There are multiple instruments present in the '
                         'spectbls.')
    return instruments[0]

def __same_star(spectbls):
    stars = np.array([s.meta['star'] for s in spectbls])
    if any(stars[1:] != stars[:-1]):
        raise ValueError('More than one target in the provided spectra.')
    return stars[0]