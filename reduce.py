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
import os.path as path

def panspectrum(specfiles, R=1000.0):
    """
    Coadd and splice the provided spectra into one panchromatic spectrum
    sampled at the native resolutions and constant R.
    """
    stars = [path.basename(s)[16:21] for s in specfiles]
    star = np.unique(stars)
    if len(star) > 1:
        raise ValueError('More than one target in the provided spectra.')
    star = star[0]
        
    #coadd all spectra from the same configuration
    allconfigs = [path.basename(s)[:15] for s in specfiles]
    configs = np.unique(allconfigs)
    coadds = []
    for config in configs:
        cfiles = filter(lambda s: config in s, specfiles)
        if len(cfiles) == 1:
            spectbls = fits2tbl(cfiles[0])
            coadds.extend(spectbls)
        else:
            spectbls = map(fits2tbl, cfiles)
            spectbls = zip(*spectbls)
            ccoadds = map(coadd, spectbls)
            coadds.append(ccoadds)
    
    #normalize the spectra
    
    
    #concatenate the coadded spectra
    catspec = concatenate(coadds)
    
    #resample at constant R
    Rspec = powerbin(catspec, R)
    
    return catspec,Rspec
    
    
def concatenate(spectra):
    """
    Concatenate spectra contained in tables or FITS files.
    
    Where spectra overlap, uses the one with the higher overall S/N.
    """
    
    if type(spectra) is str:
        #use my file naming convention to determine the configuration
        configs = [path.basename(s)[3:16] for s in spectra]
        if len(np.unique(configs)) < configs:
            warnings.warn('Two spectra from the same configurations were input. '
                          'This function will not coadd them. You may want to do '
                          'that before feeding them in here.')
        
        #read in all of the spectra and sort them by starting wavelength
        spectbls = []
        for sf in spectra:
            spectbls.extend(fits2tbl(sf))
            
    sortkey = lambda spec: np.min(spec['w0'])
    spectbls.sort(key=sortkey)
        
    #now splice them all together
    return reduce(splice, spectbls)

def normalize(spectbla, spectblb, SNcut=2.0, method='area'):
    """
    Normalized the spectrum b to spectrum a. 
    
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

def splice(spectbla, spectblb, normalize=False):
    """
    Combine two (presumably overlapping) spectra.
    
    One or two wavelengths are chosen for the switch from one spectrum to
    limit pixel-to-pixel differences. These switches are determined such that
    overall S/N is maximized. If the spectra do not overlap, they are merely
    stacked in order.
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
        ospecs[not base] = rebin(ospecs[not base], basegrid) #rebin the finer spectrum
        
        #now figure out where to splice the two
        def integratedSN(spectbls): #determine the cumulative signal to noise, summing bins
            np.vstack(map(np.array, spectbls))
            dw = abs(spectbls['w1'] - spectbls['w0'])
            fw, ew = spectbls['flux']*dw, spectbls['error']*dw
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

def fits2tbl(specfile):
    """
    A catch-all function to read in FITS spectra from all variety of 
    instruments and provide standardized output as a list of astropy tables.
    
    The standardized filename 'w aaa bbb ccccc ... .fits, where aaa is the 
    observatory, bbb is the instrument, and ccccc is the filter/grating (w is 
    the spectral band) is used to determine how to parse the FITS file .
    
    The table has columns 'w0','w1' for the wavelength edges, 'flux', 'error',
    'exptime', and 'instrument'. The 'source' column contains a number
    identifying the instrument or model source for the data.
    
    Source Identifiers
    ----------------------
    0 : HST COS G130M
    1 : HST COS G160M
    2 : HST COS G230L
    3 : HST STIS E140M
    4 : HST STIS E230M
    5 : HST STIS E230H
    6 : HST STIS G140M
    7 : HST STIS G230L
    8 : HST STIS G430L
    9 : XMM ---- -----
    """
    sourcelist = ['HST_COS_G130M','HST_COS_G160M','HST_COS_G230L',
                  'HST_STIS_E140M','HST_STIS_E230M','HST_STIS_E230H',
                  'HST_STIS_G140M','HST_STIS_G230L','HST_STIS_G430L',
                  'XMM_----_-----']
    config = path.basename(specfile)[2:15]
    index = sourcelist.index(config)
    observatory = config[:3].upper()
    
    if observatory == 'HST':
        spec = fits.open('specfile')
        getval = lambda s: spec[1].header[s]
        Norders, Npts, exptime = map(getval, ['naxi2', 'naxis1', 'exptime'])
        xnames = ['wavelength','flux','error', 'dq']
        wmid, flux, err, flags = [spec[1].data[s] for s in xnames]
        wedges = np.array(map(mnp.mids2edges, wmid))
        w0, w1 = wedges[:,:-1], wedges[:,1:]
        iarr = np.ones([Norders, Npts])*index
        exptarr = np.array([Norders,Npts])*exptime
        datas = np.array([w0,w1,flux,err,exptarr,flags,iarr])
        datas.swapaxes(0,1)
    if observatory == 'XMM':
        pass
    
    cols = ['w0','w1','flux','error','exptime','flags','source']
    dtypes = ['f4']*5 + [flags.dytpe] + ['i1']
    spectbls = [Table(data, names=cols, dtype=dtypes) for data in datas]
    return spectbls
        
def coadd(spectbls):
    listify = lambda s: [spec[s] for spec in spectbls]
    cols = ['w0','w1','flux','error','exptime','flags']
    w0, w1, f, e, w, dq = map(listify, cols)
    mask = dq > 0
    we = [np.append(ww0,ww1[-1]) for ww0,ww1 in zip(w0,w1)]
    return specutils.coadd(we, f, e, w, mask)

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