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
    
    priority = ['HST COS','HST STS']
    
    #read in all of the spectra and sort them by starting wavelength
    spectra = []
    for sf in specfiles:
        spectra.extend(__standardize(sf))
    sortkey = lambda spec: np.min(spec['w0'])
    spectra.sort(key=sortkey)
    
    #prepare a table to keep the master spectrum
    names = ['w0','w1','flux','error','source','exptime']
    dtype = ['f4','f4','f4','f4','i1','f4']
    master = Table(names=names, dtype=dtype)
    
    #make some basic functions to use in determining data quality in overlaps
    argoverlap = lambda d,wover: np.logical_and(mnp.inranges(d['w0'], wover),
                                                mnp.inranges(d['w1'], wover))
#    over_res = lambda d,arg: np.median(d['w1'][arg] - d['w0'][arg])
    def over_sn(d,arg):
        d1 = d[arg]
        dw = d1['w1'] - d1['w0']
        fluence = d1['flux']*dw
        error = d1['error']*dw
        return np.sum(fluence)/np.sqrt(np.sum(error**2))
        
    def chop_end(spec, w, side):
        if side is 'left':
            i = 0
            dwnew  = spec['w1'][0] - w
            spec['w0'][0] = w
        if side is 'right':
            i = -1
            dnew = spec['w0'][-1]
            spec['w1'][-1] = w
        spec['flux'][i] *= dwnew/dwold
        spec['error'][i] *= dwnew/dwold
        
        w0, w1 = spec['w0'][i], spec['w1'][i]
        dwold = w1 - w0
        dwnew = w1 - w if side is 'left' else w - w0
        if side is 
    
    end = 0.0
    for spec in spectra:
        w0,w1,f,e = [spec[s] for s in ['w0','w1','flux','error']]
        
        bothspecs = [master, spec]
        if w0[0] < end: # if the new spectrum overlaps the old
            wover = [w0[0], min([end, w1[-1]])] #determine the range of overlap
            
            #decide which spectrum to use
            argovers = [argoverlap[d,wover] for d in bothspecs]
            res = map(over_res, zip(bothspecs, argovers))
            sn = map(over_sn, zip(bothspecs, argovers))
            use = np.argmax(sn)
            
            #add what portions of the new spectrum will be used to the master
            newmax = spec['w1'][-1]
            if not use:
                if newmax < end:
                    continue
                if newmax > end:
                    pass
            if use:
                mark = argovers[0][0]
                if newmax < end:
                    mark1 = argovers[0][-1] + 1
                    left, right = master[:mark], master[mark1:]
                    chop_end(left, spec['w0'][0], 'right')
                    chop_end(right, spec['w1'][-1], 'left')
                    master = vstack([left, spec, right])
            if use and newmax > end:
                master.remove_rows(slice(mark,None))
                chop_end(master, spec['w0'][0], 'right')
                master = vstack([master, spec])
                
            end = max([end, w1[-1]])

def normalize(spectbls, SNcut=2.0):
    """
    Normalized the spectra in specs by ensuring the median value of overlapping
    regions are the same.
    
    The spectra are assumed to be orded by quality. Thus, spectra[0] is the
    gold standard against which all others are normalized. Use only points with
    S/N greater than SNcut when computing medians.
    """
    pass

def weave(spectbl1, spectbl2):
    """
    Combine two (presumably overlapping) spectra.
    """
    #check for overlap
    both = [spec1, spec2]
    wr1, wr2 = [[s['w0'][0], s['w1'][-1]] for s in both]
    overlapping = any(mnp.inrages(wr1, wr2))
    
    if not overlapping:
        both.sort(key = lambda s: s['w0'][0])
        return vstack(both)
    else:
        #resample each to the resolution of the other
        grids = np.array([np.append(s['w0'], s['w1'][-1]) for s in both])
        def subgrid(grid1, grid2): #same spacing as grid1 over the range of grid2
            edges = np.searchsorted(grid1, grid2[[0,-1]])
            return np.concatente([grid2[[0]], grid1[slice(*edges)], grid2[[-1]]])
        subgrids = map(subgrid, [grids, grids[::-1]])
        
def powerbin(specfile, R=1000.0):
    """
    Rebin a spectrum onto a grid with constant resolving power.
    """
    pass

def __spectbl_rebin(spec, newedges):
    oldedges = np.append(spec['w0'], spec['w1'][-1])
    flux, err = specutils.rebin(newedges, oldedges, spec['flux'], spec['error'])
    newspec = Table(spec, copy=True)
    newspec['flux'], newspec['error'] = flux, error
    return newspec    

def __standardize(specfile):
    """
    A catch-all function to read in FITS spectra from all variety of 
    instruments and provide standardized output.
    """
    pass