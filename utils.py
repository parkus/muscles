# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 18:02:03 2014

@author: Parke
"""
from _warnings import warn
from math import ceil, log10, sqrt
import rc, io, db
import mypy.my_numpy as mnp
from mypy import specutils
import numpy as np
from astropy.table import Table, Column
from astropy.table import vstack as tblstack
from os import path

keys = ['units', 'dtypes', 'fmts', 'descriptions', 'colnames']
spectbl_format = [rc.spectbl_format[key] for key in keys]
units, dtypes, fmts, descriptions, colnames = spectbl_format


def fancyBin(spec, maxpow=30000):
    """Rebin a spectrum to a coarser resolving power, but only where this actually makes it coarser."""
    pieces = []
    while len(spec) > 0:
        i = spec['instrument'][0]
        insti = (spec['instrument'] == i)
        _, ends = mnp.block_edges(insti)
        split = ends[0]
        piece = spec[:split]
        spec = spec[split:]
        w = (piece['w0'] + piece['w1']) / 2.0
        dw = (piece['w1'] - piece['w0'])
        Rs = w / dw
        if Rs.min() > maxpow:
            piece = powerbin(piece, maxpow, keep_remainder='fat')
        pieces.append(piece)
    return vstack(pieces)


def mag(spectbl, band='J'):
    w = (spectbl['w0'] + spectbl['w1']) / 2.0
    f = spectbl['flux']

    files = {'J':'2MASSJ.txt', 'H':'2MASSH.txt', 'K':'2MASSKs.txt'}

    filterfile = path.join(rc.filterpath, files[band])
    with open(filterfile) as filter:
        zeropoint = float(filter.readline().strip())
        wf, yf = np.loadtxt(filter).T

    ff = np.interp(wf, w, f)
    filterflux = np.trapz(ff*yf, wf)

    mag = -2.5*np.log10(filterflux) + zeropoint
    return mag


def flux_integral(spectbl, wa=None, wb=None, normed=False):
    """Compute integral of flux from spectbl values. Result will be in erg/s/cm2."""
    if normed:
        if 'normflux' not in spectbl.colnames:
            spectbl = add_normflux(spectbl)

    if wa is not None:
        spectbl = split_exact(spectbl, wa, 'red')
    if wb is not None:
        spectbl = split_exact(spectbl, wb, 'blue')

    dw = spectbl['w1'] - spectbl['w0']

    if normed:
        return np.sum(spectbl['normflux'] * dw)
    else:
        return np.sum(spectbl['flux'] * dw), mnp.quadsum(spectbl['error'] * dw)


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

def specwhere(spec, w):
    """Determine the splice locations where the wavelengths in w fall in the
    spectrum. Return a vecotr of flags specifying whether each w is outside the
    spectrum (-1), in the spectrum (1), in a gap in the spectrum (-2), or
    exactly on the edge of a bin in the spectrum (2)

    w must be sorted

    returns flags, indices
    indices of pts in gaps or on bin edges will be the splice index of the gap/edge
    indices of pts out of the spectral range will be 0 or len(spec)

    if w is scalar flags and indices are converted to scalar
    """
    scalar = np.isscalar(w)
    ww = np.array(w)

    # create edge vector from spectrum
    edges, jgaps = gappyedges(spec)

    # find where w values fit in that grid
    i = np.array(np.searchsorted(edges, ww, side='left'))

    # find w values that are in a gap or on a bin edge
    # those on an edge are given an index left of that edge, so can be misplaced
    ir = np.searchsorted(edges, ww, side='right')
    inspecrange = ((i > 0) & (i < len(edges)))
    on_edge = (i + 1 == ir)
    if len(jgaps):
        ingap = reduce(np.logical_or, [i == j for j in jgaps])
    else:
        ingap = np.zeros(ww.shape, bool)
    inbins = ~(ingap | on_edge) & inspecrange
    out = ~(ingap | on_edge | inspecrange)

    # flag w values
    flags = np.zeros(ww.shape, 'i1')
    flags[inbins] = 1
    flags[out] = -1
    flags[on_edge] = 2
    flags[ingap] = -2

    # shift indices to account for gaps
    while len(jgaps):
        j = jgaps[0]
        i[i >= j] -= 1
        jgaps = jgaps[1:] - 1
    i[inbins] -= 1
    i[i > len(spec)] = len(spec)

    if scalar:
        return int(flags), int(i)
    else:
        return flags, i

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


def vecs2spectbl(w0, w1, flux, err=0.0, exptime=0.0, flags=0, instrument=-99,
                 normfac=1.0, start=0.0, end=0.0, star='', filename='',
                 name='', sourcespecs=[], comments=[]):
    """
    Assemble the vector data into the standard MUSCLES spectbl format.

    Parameters
    ----------
    w0, w1, flux, err : 1-D array-like
    exptime, flags, instrument : 1-D array-like or scalar
    star : str
    sourcespecs : list of strings

    Returns
    -------
    spectbl : MUSCLES spectrum (astropy) table
    """
    datalist = [w0, w1, flux, err, exptime, flags, instrument, normfac, start,
                end]
    return list2spectbl(datalist, star, filename, name, sourcespecs, comments)


def list2spectbl(datalist, star='', filename='', name='', sourcespecs=[],
                 comments=[]):
    """
    Assemble the vector data into the standard MUSCLES spectbl format.

    Parameters
    ----------
    datalist : 7xN array-like
        rows are w0, w1, flux, err, exptime, flags, instrument all of length N
    star : str

    sourcespecs : list of strings

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
    if filename != '':
        if name == '':
            name = db.parse_name(filename)
        if star == '':
            star = db.parse_star(filename)
    meta = {'FILENAME' : filename,
            'SOURCESPECS' : sourcespecs,
            'STAR' : star,
            'NAME' : name,
            'COMMENT' : comments}
    return Table(cols, meta=meta)

def vstack(spectbls, name=''):
    stars = [s.meta['STAR'] for s in spectbls]
    if len(set(stars)) > 1:
        raise ValueError("Don't try to stack tables from different stars.")
    else:
        star = stars[0]

    spectbls = filter(lambda s: len(s) > 0, spectbls)

    getbeg = lambda s: s['w0'][0]
    getend = lambda s: s['w1'][-1]
    begs, ends = np.array(map(getbeg, spectbls)), np.array(map(getend, spectbls))
    assert np.all(begs[1:] >= ends[:-1])

    sourcespecs = []
    comments = []
    for s in spectbls:
        sourcespecs.extend(s.meta['SOURCESPECS'])
        comments.extend(s.meta['COMMENT'])
    sourcespecs = list(set(sourcespecs))
    comments = list(set(comments))

    data = []
    for colname in colnames:
        data.append(np.concatenate([s[colname] for s in spectbls]))

    return list2spectbl(data, star, name=name, sourcespecs=sourcespecs, comments=comments)

def gapsplit(spec_or_bins):
    w0, w1 = __getw0w1(spec_or_bins)
    if w0 is None:
        return spec_or_bins
    gaps = (w0[1:] > w1[:-1])
    isplit = list(np.nonzero(gaps)[0] + 1)
    isplit.insert(0, 0)
    isplit.append(None)
    return [spec_or_bins[i0:i1] for i0,i1 in zip(isplit[:-1], isplit[1:])]

def overlapping(spec_or_bins_a, spec_or_bins_b):
    """Check if there is any overlap."""
    getbins = lambda sob: sob if type(sob) is np.ndarray else wbins(sob)
    wbinsa, wbinsb = map(getbins, [spec_or_bins_a, spec_or_bins_b])
    ainb0, ainb1 = [mnp.inranges(w, wbinsb, [0, 0]) for w in wbinsa.T]
    bina0, bina1 = [mnp.inranges(w, wbinsa, [0, 0]) for w in wbinsb.T]
    return np.any(ainb0 | ainb1) or np.any(bina0 | bina1)

def overlap_ranges(spec_or_bins_a, spec_or_bins_b,):
    """Find the ranges over which two spectbls overlap."""
    ar, br = map(gapless_ranges, [spec_or_bins_a, spec_or_bins_b])
    return mnp.range_intersect(ar, br)

def keepranges(spectbl, *args, **kwargs):
    """Returns a table with only bins that are fully within the wavelength
    ranges. *args can either be a Nx2 array of ranges or w0, w1. **kwargs
    just for ends={'tight'|'loose'}"""
    keep = argrange(spectbl, *args, **kwargs)
    return spectbl[keep]

def exportrange(spectbl, w0, w1, folder, overwrite=False):
    piece = keepranges(spectbl, w0, w1)
    name = path.basename(spectbl.meta['FILENAME'])
    name = name.split('.')[0]
    name += '_waverange {}-{}.fits'.format(w0,w1)
    name = path.join(folder, name)
    io.writefits(piece, name, overwrite=overwrite)

def argoverlap(spec_or_bins_a, spec_or_bins_b, method='tight'):
    """ Find the (boolean) indices of the overlap of spectbla within spectblb
    and the reverse."""
    oranges = overlap_ranges(spec_or_bins_a, spec_or_bins_b)

    ao, bo = [argrange(s, oranges, ends=method) for s in
              [spec_or_bins_a, spec_or_bins_b]]

    return ao, bo

def argrange(spec_or_bins, *args, **kwargs):
    """Return the boolean indices of the desired range."""
    w0, w1 = __getw0w1(spec_or_bins)
    if w0 is None:
        return np.empty(0)

    if 'ends' in kwargs:
        ends = kwargs['ends']
    else:
        ends = 'tight'
    wranges = args[0] if len(args) == 1 else args
    wranges = np.array(wranges)
    wranges = np.reshape(wranges, [wranges.size / 2, 2])

    inrange = np.zeros(len(spec_or_bins), bool)
    for wr in wranges:
        if ends == 'loose':
            in0, in1 = w0 < wr[1], w1 > wr[0]
            inrange = inrange | (in0 & in1)
        if ends == 'tight':
            in0, in1 = w0 >= wr[0], w1 <= wr[1]
            inrange = inrange | (in0 & in1)

    return inrange

def gappyedges(spectbl):
    """Create a vector of bin edges with gaps included. Return the vector of
    edges and the splice indices of the gaps."""
    w0, w1 = spectbl['w0'], spectbl['w1']
    igaps = np.nonzero(w0[1:] > w1[:-1])[0] + 1
    edges = np.append(w0, w1[-1])
    edges = np.insert(edges, igaps, w1[igaps - 1])
    igaps = igaps + np.arange(len(igaps)) + 1
    return edges, igaps

def gapranges(spectbl):
    """Return a list of the ranges of the gaps in the spectrum."""
    w0, w1 = spectbl['w0'], spectbl['w1']
    igaps = np.nonzero(w0[1:] > w1[:-1])[0]
    return np.array([[w1[i], w0[i+1]] for i in igaps])

def fillgaps(spectbl, fill_value=np.nan):
    """Fill any gaps in the wavelength range of spectbl with some value."""
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

def gapless_ranges(spec_or_bins):
    w0, w1 = __getw0w1(spec_or_bins)
    if w0 is None:
        return np.empty([0,2])
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

def __getw0w1(spec_or_bins):
    if len(spec_or_bins) == 0:
        return None, None
    if type(spec_or_bins) is np.ndarray:
        w0, w1 = spec_or_bins.T
    else:
        w0, w1 = wbins(spec_or_bins).T
    return w0, w1


def rebin(spec, newbins):
    """Rebin the spectrum, dealing with gaps in newbins if appropriate."""

    # get overlapping bins, warn if some don't overlap
    _, overnew = argoverlap(spec, newbins, method='tight')
    Nkeep = np.sum(overnew)
    if Nkeep == 0:
        warn('All newbins fall outside of spec. Returning empty spectrum.')
        return spec[0:0]
    if Nkeep < len(newbins):
        warn('Some newbins fall outside of spec and will be discarded.')
    newbins = newbins[overnew]

    # split at gaps and rebin. no bins covering a gap in spec should remain in
    # newbins, so there shouldn't be a need to split newgaps
    splitbins = gapsplit(newbins)
    if len(splitbins) > 1:
        specs = []
        for bins in splitbins:
            trim = keepranges(spec, bins[0, 0], bins[-1, 1], ends='loose')
            specs.append(rebin(trim, bins))
        return vstack(specs)

    # trim down spec to avoid gaps (gaps are handled in code block above)
    spec = keepranges(spec, newbins[0, 0], newbins[-1, 1], ends='loose')

    # rebin
    w0, w1 = newbins.T
    newedges = bins2edges(newbins)
    oldedges = wedges(spec)
    dwnew, dwold = map(np.diff, [newedges, oldedges])
    flux, error, flags = specutils.rebin(newedges, oldedges, spec['flux'],
                                         spec['error'], spec['flags'])
    insts = mnp.rebin(newedges, oldedges, spec['instrument'], 'or')
    normfac = mnp.rebin(newedges, oldedges, spec['normfac'], 'avg')
    start = mnp.rebin(newedges, oldedges, spec['minobsdate'], 'min')
    end = mnp.rebin(newedges, oldedges, spec['maxobsdate'], 'max')
    expt = mnp.rebin(newedges, oldedges, spec['exptime'], 'avg')

    # spectbl accoutrments
    star, name, fn, sf = [spec.meta[s] for s in
                          ['STAR', 'NAME', 'FILENAME', 'SOURCESPECS']]

    return vecs2spectbl(w0, w1, flux, error, expt, flags, insts, normfac,
                              start, end, star, fn, name, sf)


def evenbin(spectbl, dw, lo=None, hi=None):
    if lo is None: lo = np.min(spectbl['w0'])
    if hi is None: hi = np.max(spectbl['w1'])
    newedges = np.arange(lo, hi, dw)
    newbins = edges2bins(newedges)
    return rebin(spectbl, newbins)


def powerbin(spectbl, R=1000.0, lo=None, hi=None, keep_remainder=False):
    """
    Rebin a spectrum onto a grid with constant resolving power.

    If keep_remainder is False and the constant R grid does not permit an integer number of bins to exactly cover the
    original wavelength range, a fractional bin at the end of the range will be discarded. Otherwise it can be set to
    fat or skinny to specify whether the fraction bin should be combined with the previous or kept as a fractional bin.
    """
    start = spectbl['w0'][0]
    if lo is None and start == 0:
        start = 1.0
    if lo is not None and start < lo:
        start = lo
    end = spectbl['w1'][-1] if hi is None else hi
    fac = (2.0 * R + 1.0) / (2.0 * R - 1.0)
    maxpow = ceil(log10(end / start) / log10(fac)) + 1
    powers = np.arange(maxpow)
    we = start * fac ** powers
    if keep_remainder == 'skinny':
        we[-1] = end
    elif keep_remainder == 'fat':
        we[-2] = end
        we = we[:-1]
    else:
        we = we[:-1]
    if len(we) < 2:
        we = np.array([start, end])
    wbins = edges2bins(we)
    return rebin(spectbl, wbins)


def split_exact(spectbl, w, keepside):
    """
    Split a spectrum at exactly the specified wavelength, dealing with new
    fractional bins by augmenting the error according to Poisson statistics.

    Parameters
    ----------
    spectbl : muscles spectrum
    w : float
        wavelength at which to trim
    keepside : {'red'|'blue'|'both'}

    Result
    ------
    splitspecs : muscles spectrum
        one or two spectables according to the keepside setting
    """
    keepblu = (keepside in ['blue', 'both'])
    keepred = (keepside in ['red', 'both'])

    # find the index of the bin w falls in
    flag, i = utils.specwhere(spectbl, w)

    if flag == 1:
        # w is in a bin of the spectbl
        # parse out info from bin that covers w
        error = spectbl[i]['error']
        w0, w1 = spectbl['w0'][i], spectbl['w1'][i]
        dw = w1 - w0

        # make tables with modified edge bin
        if keepblu:
            if w == w0:
                bluspec = Table(spectbl[:i], copy=True)
            else:
                bluspec = Table(spectbl[:i + 1], copy=True)
                dw_new = w - w0
                error_new = error * sqrt(dw / dw_new)
                bluspec[-1]['w1'] = w
                bluspec[-1]['error'] = error_new
        if keepred:
            redspec = Table(spectbl[i:], copy=True)
            if w != w0:
                dw_new = w1 - w
                error_new = error * sqrt(dw / dw_new)
                redspec[0]['w0'] = w
                redspec[0]['error'] = error_new
    else:
        # w is outside of the spectbl, in a gap, or right on a bin edge,
        # then i works as a slice
        if keepblu:
            bluspec = Table(spectbl[:i], copy=True)
        if keepred:
            redspec = Table(spectbl[i:], copy=True)

    if keepblu: assert np.all(bluspec['w1'] > bluspec['w0'])
    if keepred: assert np.all(redspec['w1'] > redspec['w0'])

    if keepside == 'blue':
        return bluspec
    if keepside == 'red':
        return redspec
    if keepside == 'both':
        return bluspec, redspec