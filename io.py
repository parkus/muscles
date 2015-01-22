# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:28:07 2014

@author: Parke
"""

from os import path
from astropy.io import fits
import numpy as np
from mypy.my_numpy import mids2edges
from scipy.io import readsav as spreadsav
import database as db
import utils
from astropy.table import Table

phoenixbase = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'

def read(specfile):
    """A catch-all function to read in FITS spectra from all variety of 
    instruments and provide standardized output as a list of astropy tables.
    
    The standardized filename 'w_aaa_bbb_ccccc_..._.fits, where aaa is the 
    observatory (mod used for modeled data), bbb is the instrument (or model 
    type), and ccccc is the filter/grating (w is 
    the spectral band) is used to determine how to parse the FITS file .
    
    The table has columns 'w0','w1' for the wavelength edges, 'flux', 'error',
    'exptime', 'flags', and 'source'. The 'source' column contains a number, where
    muscles.instruments[source_number] gives the aaa_bbb_ccccc string identifying
    the instrument.
    """
    
    readfunc = {'fits' : readfits, 'txt' : readtxt, 'sav' : readsav}
    i = specfile[::-1].find('.')
    fmt = specfile[-i:]
    return readfunc[fmt](specfile)

def readstdfits(specfile):
    """Read a fits file that was created by writefits."""
    spectbl = Table.read(specfile, hdu=1)
    spectbl.meta['FILENAME'] = specfile
    try:
        sourcefiles = fits.getdata(specfile, 'sourcefiles')['sourcefiles']
        spectbl.meta['SOURCEFILES'] = sourcefiles
    except KeyError:
        spectbl.meta['SOURCEFILES'] = []
    
    spectbl = utils.conform_spectbl(spectbl)
    
    return spectbl
    
def readfits(specfile):
    """Read a fits file into standardized table."""
    
    insti = db.getinsti(specfile)
    inststr = db.instruments[insti]
    observatory = inststr.split('_')[0].lower()
    
    spec = fits.open(specfile)
    if any([s in specfile for s in ['coadd', 'custom', 'mod']]):
        return [readstdfits(specfile)]
    elif observatory == 'hst':
        sd, sh = spec[1].data, spec[1].header
        flux, err = sd['flux'], sd['error']
        shape = flux.shape
        iarr = np.ones(shape)*insti
        exptime = sh['exptime']
        wmid, flags = sd['wavelength'], sd['dq']
        wedges = np.array([mids2edges(wm, 'left', 'linear-x') for wm in wmid])
        w0, w1 = wedges[:,:-1], wedges[:,1:]
        exptarr = np.ones(shape)*exptime
        datas = np.array([w0,w1,flux,err,exptarr,flags,iarr])
        datas = datas.swapaxes(0,1)
    elif observatory == 'xmm':
        colnames = ['Wavelength', 'BinWidth', 'Flux', 'FluxError', 'Flux2']
        wmid, dw, cps, cpserr, flux = [spec[2].data[s][::-1] for s in colnames]
        #TODO: make sure to look this over regularly, as these files are likely
        #to grow and change
        wepos, weneg = (wmid[:-1] + dw[:-1]), (wmid[1:] - dw[1:])
        if any(abs(wepos - weneg) > 0.01):
            raise ValueError('There are significant gaps in the XMM spectrum'
                             '\n{}'.format(path.basename(specfile)))
        we = (wepos + weneg)/2.0
        w0 = np.insert(we, 0, wmid[0] - dw[0])
        w1 = np.append(we, wmid[-1] + dw[-1])
        fluxfac = flux/cps
        err = cpserr*fluxfac
        N = len(w0)
        expt = np.nan*np.zeros(N)
        flags = np.zeros(N, 'i1')
        source = np.ones(N)*insti
        datas = [[w0,w1,flux,err,expt,flags,source]]
    else:
        raise Exception('fits2tbl cannot parse data from the {} observatory.'.format(observatory))
    
    spec.close()
    spectbls = [__maketbl(d, specfile) for d in datas]
    return spectbls
    
def readtxt(specfile):
    """
    Reads data from text files into standardized astropy table output. 
    """
    
    if 'mod_euv' in specfile.lower():
        f = open(specfile)
        wmid, iflux = [],[]
        #move to the first line of data
        while '-- WITH A SPECTRAL BREAKDOWN OF --' not in f.next(): continue
        #read the data
        for line in f:
            wmid.append(float(line[8:22]))
            iflux.append(float(line[22:39]))
        #standardize
        we = mids2edges(wmid)
        w0, w1 = we[:-1], we[1:]
        flux = np.array(iflux)/(w1 - w0)
        N = len(flux)
        err = np.ones(N)*np.nan
        expt,flags = np.zeros(N), np.zeros(N,'i1')
        source = db.getinsti(specfile)*np.ones(N)
        return [__maketbl([w0,w1,flux,err,expt,flags,source], specfile)]
    else:
        raise Exception('A parser for {} files has not been implemented.'.format(specfile[2:9]))
        
def readsav(specfile):
    """
    Reads data from IDL sav files into standardized astropy table output. 
    """
    sav = spreadsav(specfile)
    wmid = sav['w140']
    flux = sav['lya_mod']
#    velocity_range = 300 #km/s
#    wrange = velocity_range/3e5*1215.67*np.array([-1,1]) + 1215.67
#    keep = np.digitize(wmid, wrange) == 1
#    wmid, flux = wmid[keep], flux[keep]
    we = mids2edges(wmid, 'left', 'linear-x')
    w0, w1 = we[:-1], we[1:]
    N = len(flux)
    err = np.ones(N)*np.nan
    expt,flags = np.zeros(N), np.zeros(N, 'i1')
    source = db.getinsti(specfile)*np.ones(N)
    return [__maketbl([w0,w1,flux,err,expt,flags,source], specfile)]
    
def writefits(spectbl, name, overwrite=False):
    """
    Writes spectbls to a standardized MUSCLES FITS file format.
    (Or, rather, this function defines the standard.)
    
    Parameters
    ----------
    spectbl : astropy table in MUSCLES format
        The spectrum to be written to a MSUCLES FITS file. 
    name : str
        filename for the FITS output
    overwrite : {True|False}
        whether to overwrite if output file already exists
        
    Returns
    -------
    None
    """
    #use the native astropy function to write to fits
    try:
        sourcefiles = spectbl.meta['SOURCEFILES']
        del spectbl.meta['SOURCEFILES'] #otherwise this makes a giant nasty header
    except KeyError:
        sourcefiles = [spectbl.meta['FILENAME']]
    spectbl.meta['FILENAME'] = name
    spectbl.write(name, overwrite=overwrite, format='fits')
    
    #but open it up to do some modification
    with fits.open(name, mode='update') as ftbl:
        
        #add name to first table
        ftbl[1].name = 'spectrum'
        
        #add column descriptions
        for i,name in enumerate(spectbl.colnames):
            key = 'TDESC' + str(i+1)
            ftbl[1].header[key] = spectbl[name].description
            
        #add an extra bintable for the instrument identifiers
        col = [fits.Column('instruments','13A',array=db.instruments)]
        hdr = fits.Header()
        hdr['comment'] = ('This extension is a legend for the integer '
                          'identifiers in the instrument '
                          'column of the previous extension.')
        idhdu = fits.BinTableHDU.from_columns(col, header=hdr, name='legend')
        ftbl.append(idhdu)
        
        #add another bintable for the sourcefiles, if needed
        if len(sourcefiles):
            maxlen = max([len(sf) for sf in sourcefiles])
            dtype = '{:d}A'.format(maxlen)
            col = [fits.Column('sourcefiles', dtype, array=sourcefiles)]
            hdr = fits.Header()
            hdr['comment'] = ('This extension contains a list of the source '
                              'files that were incorporated into this '
                              'spectrum.')
            sfhdu = fits.BinTableHDU.from_columns(col, header=hdr, 
                                                  name='sourcefiles')
            ftbl.append(sfhdu)
        
        ftbl.flush()
        
def phxpath(Teff, logg=4.5, FeH=0.0, aM=0.0, repo='ftp'):
    """
    Constructs the URL for the phoenix spectrum file for a star with effective
    temperature Teff, log surface gravity logg, metalicity FeH, and alpha
    elemnt abundance aM.
    
    Does not check that the URL is actually valid, and digits beyond the
    precision of the numbers used in the path will be truncated.
    """
    zstr = '{:+4.1f}'.format(FeH)
    if FeH == 0.0: zstr = '-' + zstr[1:]
    astr = '.Alpha={:+5.2f}'.format(aM) if aM != 0.0 else ''
    name = ('lte{T:05.0f}-{g:4.2f}{z}{a}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
            ''.format(T=Teff, g=logg, z=zstr, a=astr))
    
    if repo == 'ftp':
        folder = 'Z' + zstr + astr + '/'
        return phoenixbase + folder + name
    else:
        return path.join(repo, name)
    
def phxdata(Teff, logg=4.5, FeH=0.0, aM=0.0, repo='ftp'):
    """
    Get a phoenix spectral data from the repo and return as an array.
    """
    path = phxpath(Teff, logg, FeH, aM, repo=repo)
    fspec = fits.open(path)
    return fspec[0].data

def __maketbl(data, specfile, sourcefiles=[]):
    star = specfile.split('_')[4]
    return utils.list2spectbl(data, star, specfile, sourcefiles)
    