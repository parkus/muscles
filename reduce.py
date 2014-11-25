# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import numpy as np
from astropy.table import Table, vstack
import my_numpy as mnp
from math import sqrt
import specutils
from muscles.database import instruments
from muscles import utils

def panspectrum(spectbls, R=1000.0):
    """
    Coadd and splice the provided spectra into one panchromatic spectrum
    sampled at the native resolutions and constant R.
    
    Overlapping spectra will be normalized with the assumptions that they are 
    listed in order of descending quality. 
    """
    #make sure all spectra are of the same star
    star = __same_star(spectbls)
    
    #make sure spectra are each from a single source
    for i,s in enumerate(spectbls):
        try: 
            __same_inst([s])['instrument']
        except ValueError:
            raise ValueError('More than one instrument used in spectbl {}'.format(i))
    
    #coadd all spectra from the same configuration
    allconfigs = [s['instrument'][0] for s in spectbls]
    configs = np.unique(allconfigs)
    coadds = []
    for config in configs:
        ctbls = filter(lambda tbl: all(tbl['instrument'] == config), spectbls)
        if len(ctbls) == 1:
            coadds.extend(ctbls)
        else:
            coadds.append(coadd(ctbls))
    
    #parse the modeled from the observed spectra
    models = [i for i in range(len(instruments)) if 'mod' in instruments[i]]
    ismodel = lambda spec: spec['instrument'][0] in models
    modelspecs = filter(ismodel, coadds)
    obsspecs = filter(lambda spec: not ismodel(spec), coadds)
    
    #normalize the measured spectra
    #run through all possible pairs to see if they overlap. If so, normalize
    #according to their input order
    N = len(obsspecs)
    for i in range(N):
        for j in np.arange(i+1,N):
            if __overlapping(obsspecs[i], obsspecs[j]):
                posi,posj = [allconfigs.index(configs[k]) for k in [i,j]]
                if posi < posj: obsspecs[j] = normalize(obsspecs[i], obsspecs[j])
                if posj < posi: obsspecs[i] = normalize(obsspecs[j], obsspecs[i])
    
    
    #splice together all the measured spectra based on S/N
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
    over = __argoverlap(both)
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
        return vstack(both)
    else: #they overlap
        #resample to the finer spectrum to the resolution of the coarser
        #figure out the (boolean) indices of the overlap
        over = __argoverlap(both)
        #count the number of bin edges in the overlap to quanitfy resolution
        res = map(np.sum, over)
        #determine which is the lower resolution to use as the base for comparison
        base = np.argmin(res)
        #confine spectra to just the overlap, rebinning the higher res spectrum
        w0base, w1base = [both[base][s][over[base]] for s in ['w0', 'w1']]
        basegrid = np.append(w0base, w1base[-1])
        N = len(basegrid) - 1
        ospecs = [b[o] for b,o in zip(both, over)] #trim spectra to just overlap
        ospecs[not base] = rebin(ospecs[not base], basegrid) #rebin the finer spectrum
        
        #now figure out where to splice the two
        def integratedSN(spectbls): #determine the cumulative signal to noise, summing bins
            np.vstack(map(np.array, spectbls))
            dw = abs(spectbls['w1'] - spectbls['w0'])
            fw, ew = spectbls['flux']*dw, spectbls['error']*dw
            return np.sum(fw, 1)/np.sqrt(np.sum(ew**2), 1)
        if wrb[1] > wra[1]: #just the ends of the spectra overlap
            #computed integrated SN for all possible splice positions, choose the best
            splicespecs = [vstack([ospecs[0][:i], ospecs[1][i:]]) 
                           for i in range(N)]
            fullSN = integratedSN(splicespecs)
            splice = np.argmax(fullSN)
            globsplices = [np.nonzero(o)[splice] for o in over]
            leftspec = both[0][:globsplices[0]]
            rightspec = both[1][globsplices[1:]]
            return vstack([leftspec, rightspec])
        else: #both[1] is fully within both[0]
            #look for a single block of both[1] to put into both[0] that
            #maximizes the SN
            SN = [np.array(s['flux']/s['error']) for s in ospecs]
            if all(SN[0] > SN[1]): #don't use ospec[1] at all
                return both[0]
            beg = np.argmax(SN[1]/SN[0]) #start where the SN of ospec[1] is best
            leftblocks = [vstack([ospecs[0][:i0], ospecs[1][i0:beg], ospecs[0][beg:]])
                          for i0 in np.arange(0,beg+1)]
            i0 = np.argmax(integratedSN(leftblocks))
            rightblocks = [vstack([ospecs[0][:i0], ospecs[1][i0:i1], ospecs[0][:i1]])
                           for i1 in np.arange(beg,N)]
            i1 = np.argmax(integratedSN(rightblocks))
            globi0 = [np.nonzero(o)[i0] for o in over]
            globi1 = [np.nonzero(o)[i1] for o in over]
            leftspec = both[0][:globi0[0]]
            midspec = both[1][globi0[0]:globi1[1]]
            rightspec = both[0][globi1[0]:]
            return vstack([leftspec, midspec, rightspec])

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
    def cutoff(specrow, wnew, newside):
        dwold = specrow['w1'] - specrow['w0']
        if newside == 0:
            dwnew = specrow['w1'] - wnew
            specrow['w0'] = wnew
        else:
            dwnew = wnew - specrow['w0']
            specrow['w1'] = wnew
        specrow['error'] *= sqrt(dwold/dwnew)
    
    #determine where the endpts of spectblb fall in spectbla
    wedges_a = np.append(spectbla['w0'], spectbla['w1'][-1])
    wrange_b = [spectbla['w0'][0], spectbla['w1'][-1]]
    args = np.searchsorted(wedges_a, wrange_b)
    
    #deal with overlap
    speclist = []
    if args[0] == Na: #the left side of b is right of a
        speclist.extend([spectbla, spectblb])
    elif args[0] > 0: #the left side of b is in a
        leftspec = spectbla[args[0]-1:]
        cutoff(leftspec[1], spectblb[0]['w0'], 1)
        speclist.extend([leftspec,spectblb])
    else: #the left side of b is left of a
        speclist.append(spectblb)
    if args[1] == 0: #if the right side of b is left of a
        speclist.append(spectbla)
    if args[1] < Na: #if the right side of b is in a
        rightspec = spectbla[:args[1]-1]
        cutoff(rightspec[0], spectblb[-1]['w1'], 0)
        speclist.append(rightspec)
    
    return vstack(speclist)

def powerbin(spectbl, R=1000.0):
    """
    Rebin a spectrum onto a grid with constant resolving power.
    
    If the constant R grid cannot does not permit an integer number of bins
    within the original wavelength range, the remainder will be discarded.
    """
    start = spectbl['w0'][0]
    end = spectbl['w1'][-1]
    dwmin = start/R
    Nmax = (end-start)//dwmin + 1
    w = np.zeros(Nmax)
    w[0] = start
    for i in np.arange(1,Nmax): w[i] = w[i-1]*(1 + 1/(R - 0.5))
    w = w[w <= end]
    return rebin(spectbl, w)
        
def coadd(spectbls, maskbaddata=True):
    """Coadd spectra in spectbls."""
    inst = __same_instrument(spectbls)
    star = __same_star(spectbls)
    
    sourcefiles = []
    for s in spectbls: sourcefiles.extend(s.meta['sourcefiles'])
    
    listify = lambda s: [spec[s] for spec in spectbls]
    cols = ['w0','w1','flux','error','exptime','flags']
    w0, w1, f, e, expt, dq = map(listify, cols)
    we = [np.append(ww0,ww1[-1]) for ww0,ww1 in zip(w0,w1)]
    if maskbaddata:
        mask = dq > 0
        cwe, cf, ce, cexpt = specutils.coadd(we, f, e, expt, mask)
        cw0, cw1 = we[:-1], we[1:]
        goodbins = (cexpt > 0)
        cw0,cw1,cf,ce,cexpt = [v[goodbins] for v in [cw0, cw1, cf, ce, cexpt]]
        dq = 0
    else:
        cwe, cf, ce, cexpt = specutils.coadd(we, f, e, expt)
        cw0, cw1 = we[:-1], we[1:]
        dq = np.nan
        
    return utils.vecs2spectbl(cw0,cw1,cf,ce,cexpt,dq,inst,star,sourcefiles)

def rebin(spec, newedges):
    oldedges = np.append(spec['w0'], spec['w1'][-1])
    flux, error = specutils.rebin(newedges, oldedges, spec['flux'], spec['error'])
    newspec = Table(spec, copy=True)
    newspec['flux'], newspec['error'] = flux, error
    return newspec    

def __argoverlap(spectbl0, spectbl1):
    """ Find the (boolean) indices of the overlap of spectbl0 within spectbl1
    and the reverse.
    """
    both = [spectbl0, spectbl1]
    grids = [np.append(s['w0'], s['w1'][-1]) for s in both]
    over0w1 = mnp.inranges(grids[0], grids[1][[0,-1]])
    over1w0 = mnp.inranges(grids[1], grids[0][[0,-1]])
    return over0w1, over1w0
    
def __overlapping(spectbl0, spectbl1):
    """Check if they overlap."""
    wr = lambda spec: [spec['w0'][0], spec['w1'][1]]
    return any(np.digitize(wr(spectbl0), wr(spectbl1)) == 1)
    
def __same_instrument(spectbls):
    instruments = []
    for s in spectbls: instruments.extend(s)
    if any(instruments[:-1] != instruments[1:]):
        raise ValueError('There are multiple instruments present in the '
                         'spectbls.')
    return instruments[0]

def __same_star(spectbls):
    stars = [s.meta['star'] for s in spectbls]
    if any(stars[1:] != stars[:-1]):
        raise ValueError('More than one target in the provided spectra.')
    return stars[0]