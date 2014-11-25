# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:28:07 2014

@author: Parke
"""

from os import path
from astropy.io import fits
import numpy as np
from my_numpy import mids2edges
from scipy.io import readsav as spreadsav
from database import instruments
import utils

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

def readfits(specfile):
    """Read a fits file into standardized table."""
    
    insti = __inst_i(specfile)
    observatory = insti.split('_')[0].upper()
    
    if observatory == 'HST':
        spec = fits.open(specfile)
        exptime = spec[1].header['exptime']
        xnames = ['wavelength','flux','error', 'dq']
        wmid, flux, err, flags = [spec[1].data[s] for s in xnames]
        wedges = np.array([mids2edges(wm, 'left', 'linear-x') for wm in wmid])
        w0, w1 = wedges[:,:-1], wedges[:,1:]
        shape = flux.shape
        iarr = np.ones(shape)*insti
        exptarr = np.ones(shape)*exptime
        datas = np.array([w0,w1,flux,err,exptarr,flags,iarr])
        datas = datas.swapaxes(0,1)
    elif observatory == 'XMM':
        spec = fits.open(specfile)
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
        source = __inst_i(specfile)*np.ones(N)
        return __maketbl([w0,w1,flux,err,expt,flags,source], specfile),
    else:
        raise Exception('A parser for {} files has not been implemented.'.format(specfile[2:9]))
        
def readsav(specfile):
    """
    Reads data from IDL sav files into standardized astropy table output. 
    """
    sav = spreadsav(specfile)
    wmid = sav['w140']
    flux = sav['lya_mod']
    velocity_range = 200 #km/s
    wrange = velocity_range/3e5*1215.67*np.array([-1,1]) + 1215.67
    keep = np.digitize(wmid, wrange) == 1
    wmid, flux = wmid[keep], flux[keep]
    we = mids2edges(wmid, 'left', 'linear-x')
    w0, w1 = we[:-1], we[1:]
    N = len(flux)
    err = np.ones(N)*np.nan
    expt,flags = np.zeros(N), np.zeros(N, 'i1')
    source = __inst_i(specfile)*np.ones(N)
    return __maketbl([w0,w1,flux,err,expt,flags,source], specfile),
    
def writefits(spectbl, name, overwrite=False):
    #use the native astropy function to write to fits
    spectbl.write(name, overwrite=overwrite, fromat='fits')
    
    #but open it up to do some modification
    with fits.open(name, mode='update') as ftbl:
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
        #add column descriptions
        for i,name in enumerate(spectbl.colnames):
            key = 'TDESC' + str(i)
            ftbl[1].header[key] = spectbl[name].description
            
        #add an extra bintable for the instrument identifiers
        col = [fits.Column('instruments','13A',array=instruments)]
        hdr = fits.Header()
        hdr['comment'] = ('The instruments listed in this table correspond to the '
                          'identifiers in the instrument column of the spectrum '
                          'table. The number listed in that column is the index '
                          'of the row in this table.')
        idhdu = fits.BinTableHDU.from_columns(col, header=hdr)
        ftbl.append(idhdu)
        ftbl.flush()   
    
def __maketbl(data, specfile):
    star = specfile.split('_')[4]
    sourcefiles = [specfile]
    return utils.list2spectbl(data, star, sourcefiles)
    
def __inst_i(filename):
    name = path.basename(filename)
    inst = name.split('_')[1:4]
    inst_str = '_'.join(inst)
    return instruments.index(inst_str)
    