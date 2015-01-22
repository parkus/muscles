# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:02:03 2014

@author: Parke
"""

import settings
import mypy.my_numpy as mnp
import numpy as np
from astropy.table import Table, Column

keys = ['units', 'dtypes', 'fmts', 'descriptions', 'colnames']
spectbl_format = [settings.spectbl_format[key] for key in keys]
units, dtypes, fmts, descriptions, colnames = spectbl_format

def clooge_edges(mids):
    """Just uses the midpoints of the midpoints to guess at the edges for
    a grid. Taking the midpoints of the returned bin edges will _not_ reproduce
    the input."""
    edges = mnp.midpts(mids)
    beg = mids[0] - (edges[0] - mids[0])
    end = mids[-1] + (mids[-1] - edges[-1])
    return np.concatenate([[beg], edges, [end]])
    
def conform_spectbl(spectbl):
    """Make sure all columns have the appropriate data types, formats,
    descriptions, and units."""
    cols = []
    meta = spectbl.meta
    for u,dt,f,de,n in zip(*spectbl_format):
        #ugh, there is some astropy bug in the dtype casting that corrupts
        #the numbers int he column if, e.g., '>f8' is cast to 'f8'. this is
        #a workaround
        dt = np.result_type(dt, spectbl[n].dtype)
        cols.append(Column(spectbl[n], n, dt, unit=u, description=de, format=f))
    return Table(cols, meta=meta)
    
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
    return list2spectbl(datalist, star, filename, sourcefiles)

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
    
    cols = [Column(d,n,dt,description=dn,unit=u,format=f) for d,n,dt,dn,u,f in
            zip(datalist,colnames,dtypes,descriptions,units,fmts)]
    meta = {'FILENAME' : filename,
            'SOURCEFILES' : sourcefiles,
            'STAR' : star}
    return Table(cols, meta=meta)
    
def vstack(spectbls):
    stars = [s.meta['STAR'] for s in spectbls]
    if len(set(stars)) > 1:
        raise ValueError("Don't try to stack tables from different stars.")
    else:
        star = stars[0]
        
    sourcefiles = []
    for s in spectbls: sourcefiles.extend(s.meta['SOURCEFILES'])
    sourcefiles = list(set(sourcefiles))
    
    data = []
    for name in colnames:
        data.append(np.concatenate([s[name] for s in spectbls]))
        
    return list2spectbl(data, star, '', sourcefiles)
        