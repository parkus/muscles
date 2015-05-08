# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:02:03 2014

@author: Parke
"""

import settings, io
import mypy.my_numpy as mnp
import numpy as np
from astropy.table import Table, Column
from astropy.table import vstack as tblstack
from os import path

keys = ['units', 'dtypes', 'fmts', 'descriptions', 'colnames']
spectbl_format = [settings.spectbl_format[key] for key in keys]
units, dtypes, fmts, descriptions, colnames = spectbl_format

def flux_integral(spectbl):
    """Compute integral of flux from spectbl values. Result will be in erg/s/cm2."""
    dw = spectbl['w1'] - spectbl['w0']
    return np.sum(spectbl['flux'] * dw)

def bol2sol(a):
    """Convert bolometric-normalized fluxes to Earth-equivalent fluxes."""
    return a*1363100

def add_normflux(spectbl):
    """Add columns to the spectbl that are the bolometric-normalized flux and
    associated error."""
    normfac = flux_integral(spectbl)
    spectbl['normflux'] = spectbl['flux']/normfac
    spectbl['normerr'] = spectbl['error']/normfac
    return spectbl

def isechelle(str_or_spectbl):
    if type(str_or_spectbl) is str:
        name = str_or_spectbl
    else:
        name = str_or_spectbl.meta['FILENAME']
    return '_sts_e' in name

def keepranges(spectbl, *args):
    """Returns a table with only bins that are fully within the wavelength
    ranges. *args can either be a Nx2 array of ranges or w0, w1."""
    keep = argrange(spectbl, *args)
    return spectbl[keep]

def exportrange(spectbl, w0, w1, folder, overwrite=False):
    piece = keepranges(spectbl, w0, w1)
    name = path.basename(spectbl.meta['FILENAME'])
    name = name.split('.')[0]
    name += '_waverange {}-{}.fits'.format(w0,w1)
    name = path.join(folder, name)
    io.writefits(piece, name, overwrite=overwrite)

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

def vecs2spectbl(w0, w1, flux, err, exptime, flags, instrument, normfac,
                 start, end, star, filename, sourcefiles=[]):
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
    #TODO: add new 'name' meta
    datalist = [w0, w1, flux, err, exptime, flags, instrument, normfac, start, end]
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

    #expand any scalar values
    N = len(datalist[2]) #length of flux vector
    expand = lambda vec: vec if hasattr(vec, '__iter__') else np.array([vec]*N)
    datalist = map(expand, datalist)

    #make table
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

def gapsplit(spectbl):
    gaps = (spectbl['w0'][1:] > spectbl['w1'][:-1])
    isplit = list(np.nonzero(gaps)[0] + 1)
    isplit.insert(0, 0)
    isplit.append(None)
    return [spectbl[i0:i1] for i0,i1 in zip(isplit[:-1], isplit[1:])]

def overlapping(spectbla, spectblb):
    """Check if there is any overlap."""
    wbinsa, wbinsb = map(wbins, [spectbla, spectblb])
    ainb0, ainb1 = [mnp.inranges(w, wbinsb) for w in wbinsa.T]
    bina0, bina1 = [mnp.inranges(w, wbinsa) for w in wbinsb.T]
    return np.any(ainb0 | ainb1) or np.any(bina0 | bina1)

def argoverlap(spectbla, spectblb, method='tight'):
    """ Find the (boolean) indices of the overlap of spectbla within spectblb
    and the reverse."""
    if not overlapping(spectbla, spectblb):
        raise ValueError('Spectra do not overlap.')

    #make edge vectors, recording gap locations
    (ea, igapsa), (eb, igapsb) = map(gappyedges, [spectbla, spectblb])

    ao = __getoverlap(eb, ea, igapsb, igapsa, method)
    bo = __getoverlap(ea, eb, igapsa, igapsb, method)

    return ao, bo

def argrange(spectbl, *args):
    """Return the boolean indices of the desired range. Inclusive."""
    wrange = args[0] if len(args) == 1 else args
    in0, in1 = [mnp.inranges(spectbl[s], wrange, [1, 1]) for s in ['w0', 'w1']]
    return in0 & in1

def __getoverlap(ea, eb, igapsa, igapsb, method):
    """Create boolean vector of overlap of b -- for use with argoverlap."""
    #find overlap
    overlap = mnp.binoverlap(ea, eb, method=method)

    #set values to false that are within gaps of a
    g0, g1 = ea[igapsa - 1], ea[igapsa]
    gaps = np.array([g0, g1]).T

    if method == 'tight':
        in0, in1 = map(mnp.inranges, [eb[:-1], eb[1:]], [gaps]*2, [[0,0]]*2)
        ingap = (in0 | in1)
    elif method == 'loose':
        in0, in1 = map(mnp.inranges, [eb[:-1], eb[1:]], [gaps]*2, [[1,1]]*2)
        ingap = (in0 & in1)
    overlap[ingap] = False

    #delete values that represent gaps in b
    overlap = np.delete(overlap, igapsb)

    return overlap

def gappyedges(spectbl):
    """Create a vector of bin edges with gaps included. Return the vector of
    edges and the splice indices of the gaps."""
    w0, w1 = spectbl['w0'], spectbl['w1']
    igaps = np.nonzero(w0[1:] > w1[:-1])[0] + 1
    edges = np.append(w0, w1[-1])
    edges = np.insert(edges, igaps, w1[igaps - 1])
    igaps = igaps + np.arange(len(igaps)) + 1
    return edges, igaps

def fillgaps(spectbl, fill_value=np.nan):
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
    filledtbl = tblstack([spectbl, gaptbl])
    filledtbl.sort('w0')
    return filledtbl

def gapless_ranges(spectbl_or_array):
    if type(spectbl_or_array) is np.ndarray:
        w0, w1 = spectbl_or_array.T
    else:
        w0, w1 = wbins(spectbl_or_array).T
    gaps = np.nonzero(w0[1:] != w1[:-1])[0] + 1
    w0s, w1s = map(np.split, [w0, w1], [gaps, gaps])
    ranges = [[ww0[0], ww1[-1]] for ww0, ww1 in zip(w0s, w1s)]
    return np.array(ranges)

def hasgaps(spectbl):
    return np.any(spectbl['w1'][:-1] < spectbl['w0'][1:])

def edges2bins(we):
    return np.array([we[:-1], we[1:]]).T

def bins2edges(wbins):
    w0, w1 = wbins.T
    if ~np.allclose(w0[1:], w1[:-1]):
        raise ValueError('There are gaps in the spectrum.')
    else:
        return np.append(w0, w1[-1])

def wbins(spectbl):
    return np.array([spectbl['w0'], spectbl['w1']]).T

def wedges(spectbl):
    return bins2edges(wbins(spectbl))
