# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import warnings
import my_numpy as mnp
import specutils

def cat_specs(specfiles):
    """
    Concatenate spectra contained in FITS files.
    
    Where spectra overlap, uses the one with the higher overall S/N.
    """
    
    #use my file naming convention to determine the configuration
    configs = [s[3:16] for s in specfiles]
    if len(np.unique(configs)) < configs:
        warnings.warn('Two spectra from the same configurations were input. '
                      'This function will not coadd them. You may want to do '
                      'that before feeding them in here.')
    
    #read in all of the spectra and sort them by starting wavelength
    spectra = []
    for sf in specfiles:
        spectra.extend(spectbl(sf))
    sortkey = lambda spec: np.min(spec['w0'])
    spectra.sort(key=sortkey)
    
    #now splice them all together
    return reduce(splice, spectra)

def normalize(spectbla, spectblb, SNcut=2.0):
    """
    Normalized the spectra in specs by ensuring the median value of overlapping
    regions are the same.
    
    The spectra are assumed to be orded by quality. Thus, spectra[0] is the
    gold standard against which all others are normalized. Use only points with
    S/N greater than SNcut when computing medians.
    """
    both = [spectbla, spectblb]
    over = __argoverlap(both)
    ospecs = [s[o] for s,o in zip(both,over)]
    SNs = [s['flux']/s['error'] for s in ospecs]
    fluxes = [s['flux'] for s in ospecs]
    for flux,SN in zip(fluxes,SNs): flux[SN < SNcut] = np.nan
    meds = [np.nanmedian(flux) for flux in fluxes]
    normfac = meds[0]/meds[1]
    normspec = Table(spectblb, copy=True)
    normspec['flux'] *= normfac
    normspec['error'] *= normfac
    return normspec

def splice(spectbla, spectblb):
    """
    Combine two (presumably overlapping) spectra.
    
    One or two wavelengths are chosen for the switch from one spectrum to
    limit pixel-to-pixel differences. These switches are determined such that
    overall S/N is maximized.
    """
    #sort the two spectra
    both = [spectbla, spectblb]
    both.sort(key = lambda s: s['w0'][0])
    
    #check for overlap
    wr1, wr2 = [[s['w0'][0], s['w1'][-1]] for s in both]
    if wr2[0] >= wr1[1]: #they don't overlap
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
        ospecs[not base] = __spectbl_rebin(ospecs[not base], basegrid) #rebin the finer spectrum
        
        #now figure out where to splice the two
        def integratedSN(spectbls): #determine the cumulative signal to noise, summing bins
            np.vstack(map(np.array, spectbls))
            dw = abs(spectbls['w1'] - spectbls['w0'])
            fw, ew = spectbl['flux']*dw, spectbl['error']*dw
            return np.sum(fw, 1)/np.sqrt(np.sum(ew**2), 1)
        if wr2[1] > wr1[1]: #just the ends of the spectra overlap
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
        
def powerbin(specfile, R=1000.0):
    """
    Rebin a spectrum onto a grid with constant resolving power.
    """
    pass

def spectbl(specfile):
    """
    A catch-all function to read in FITS spectra from all variety of 
    instruments and provide standardized output as an astropy table.
    
    The standardized filename 'aaa bbb ccccc ... .fits, where aaa is the 
    observatory, bbb is the instrument, and ccccc is the filter/grating is used
    to determine how to parse the FITS file.
    """
    pass

def __spectbl_rebin(spec, newedges):
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