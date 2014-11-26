# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:02:03 2014

@author: Parke
"""

import my_numpy as mnp
import numpy as np
from astropy.table import Table, Column
import database as db

def clooge_edges(mids):
    """Just uses the midpoints of the midpoints to guess at the edges for
    a grid. Taking the midpoints of the returned bin edges will _not_ reproduce
    the input."""
    edges = mnp.midpts(mids)
    beg = mids[0] - (edges[0] - mids[0])
    end = mids[-1] + (mids[-1] - edges[-1])
    return np.concatenate([[beg], edges, [end]])

def group_by_instrument(spectbls):
    """Group the spectbls by instrument, returning a list of the groups. Useful
    for coaddition. Preserves order."""
    
    #get the unique instruments
    sourcefiles = [spec.meta['filename'] for spec in spectbls]
    allinstruments = np.array(map(db.parse_instrument, sourcefiles))
    instruments, ind = np.unique(allinstruments, return_index=True)
    instruments = instruments[np.argsort(ind)]
    
    #group em
    groups = []
    for inst in instruments:
        use = np.nonzero(allinstruments == inst)
        specgroup = [spectbls[i] for i in use]
        groups.append(specgroup)
    
    return groups
    
def vecs2spectbl(w0, w1, flux, err, exptime, flags, instrument, star, 
                 filename, sourcefiles=[]):
    """
    Assemble the vector data into the standard MUSCLES spectbl format.
    
    Parameters
    ----------
    w0, w1, flux, err : 1-D array-like
    exptime, flags, instrument : 1-D array-like or scalar
    star : str
    sourcefiles : list of strings
    
    Returns
    -------
    spectbl : MUSCLES spectrum (astropy) table
    """
    N = len(flux)
    expand = lambda vec: vec if hasattr(vec, '__iter__') else np.array([vec]*N)
    exptime, flags, instrument = map(expand, [exptime, flags, instrument])
    datalist = [w0, w1, flux, err, exptime, flags, instrument]
    return list2spectbl(datalist, star, sourcefiles)

def list2spectbl(datalist, star, filename, sourcefiles=[]):
    """
    Assemble the vector data into the standard MUSCLES spectbl format.
    
    Parameters
    ----------
    datalist : 7xN array-like
        rows are w0, w1, flux, err, exptime, flags, instrument all of length N
    star : str
    sourcefiles : list of strings
    
    Returns
    -------
    spectbl : MUSCLES spectrum (astropy) table
    """
    names = ['w0','w1','flux','error','exptime','flags','instrument']
    units = ['Angstrom']*2 + ['erg/s/cm2/Angstrom']*2 + ['s','','']
    dtypes = ['f8']*5 + ['i', 'i1']
    fmts = ['.2f']*2 + ['.2e']*2 + ['.1f', 'b', 'd']
    descriptions = ['left (short,blue) edge of the wavelength bin',
                    'right (long,red) edge of the wavelength bin',
                    'average flux over the bin',
                    'error on the flux',
                    'cumulative exposure time for the bin',
                    'data quality flags (specific to the instrument)',
                    'identifier for the instrument that is the source of the '
                    'data. use muscles.instruments[identifier] to determine '
                    'the instrument.']
    cols = [Column(d,n,dt,description=dn,unit=u,format=f) for d,n,dt,dn,u,f in
            zip(datalist,names,dtypes,descriptions,units,fmts)]
    meta = {'filename' : filename
            'sourcefiles' : sourcefiles,
            'star' : star}
    return Table(cols, meta=meta)