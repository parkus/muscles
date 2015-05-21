# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:28:07 2014

@author: Parke
"""

from os import path
from astropy.io import fits
import numpy as np
from mypy.my_numpy import mids2edges, block_edges
from scipy.io import readsav as spreadsav
import database as db
import utils, settings
from astropy.table import Table
from astropy.time import Time
from warnings import warn

phoenixbase = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'

def readpans(star):
    """
    Read in all panspectra for a star and return as a list.
    """
    panfiles = db.allpans(star)
    return sum(map(read, panfiles), [])

def read(specfiles):
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

    The star keyword is used to reject any spectra that are known to be bad
    for that star.
    """
    #if a list of files is provided, reach each and stack the spectra in a list
    if hasattr(specfiles, '__iter__'):
        return sum(map(read, specfiles), [])

    specfiles = db.validpath(specfiles)

    readfunc = {'fits' : readfits, 'txt' : readtxt, 'sav' : readsav}
    star = db.parse_star(specfiles)
    i = specfiles[::-1].find('.')
    fmt = specfiles[-i:]
    specs = readfunc[fmt](specfiles)
    sets = settings.load(star)
    if 'coadd' not in specfiles and 'custom' not in specfiles:
        for config, i in sets.reject_specs:
            if config in specfiles:
                specs.pop(i)
    return specs

def readstdfits(specfile):
    """Read a fits file that was created by writefits."""
    spectbl = Table.read(specfile, hdu=1)
    spectbl.meta['FILENAME'] = specfile
    spectbl.meta['NAME'] = db.parse_name(specfile)
    try:
        sourcespecs = fits.getdata(specfile, 'sourcespecs')['sourcespecs']
        spectbl.meta['SOURCESPECS'] = sourcespecs
    except KeyError:
        spectbl.meta['SOURCESPECS'] = []

    if 'hst' in specfile:
        spectbl = __trimHSTtbl(spectbl)

    spectbl = utils.conform_spectbl(spectbl)

    return spectbl

def readfits(specfile):
    """Read a fits file into standardized table."""

    observatory = db.parse_observatory(specfile)
    insti = db.getinsti(specfile)

    spec = fits.open(specfile)
    if any([s in specfile for s in ['coadd', 'custom', 'mod', 'panspec']]):
        return [readstdfits(specfile)]
    elif observatory == 'hst':
        sd, sh = spec[1].data, spec[1].header
        flux, err = sd['flux'], sd['error']
        shape = flux.shape
        iarr = np.ones(shape)*insti
        wmid, flags = sd['wavelength'], sd['dq']
        wedges = np.array([mids2edges(wm, 'left', 'linear-x') for wm in wmid])
        w0, w1 = wedges[:,:-1], wedges[:,1:]
        exptarr, start, end = [np.ones(shape)*sh[s] for s in
                               ['exptime', 'expstart', 'expend']]
        normfac = np.ones(shape)
        datas = np.array([w0,w1,flux,err,exptarr,flags,iarr,normfac,start,end])
        datas = datas.swapaxes(0,1)
        spectbls = [__maketbl(d, specfile) for d in datas]

        #cull off-detector data
        spectbls = [__trimHSTtbl(spectbl) for spectbl in spectbls]

    elif observatory == 'xmm':
        sh = spec[0].header
        dw = 5.0
        colnames = ['Wave', 'CFlux', 'CFlux_err']
        wmid, flux, err = [spec[1].data[s] for s in colnames]
        #TODO: make sure to look this over regularly, as these files are likely
        #to grow and change
        wepos, weneg = (wmid[:-1] + dw / 2.0), (wmid[1:] - dw / 2.0)
        if any(abs(wepos - weneg) > 0.01):
            raise ValueError('There are significant gaps in the XMM spectrum'
                             '\n{}'.format(path.basename(specfile)))
        # to ensure gaps aren't introduced due to slight errors...
        we = (wepos + weneg) / 2.0
        w0 = np.insert(we, 0, wmid[0] - dw)
        w1 = np.append(we, wmid[-1] + dw)

        optel = db.parse_spectrograph(specfile)
        if optel == 'pn-':
            expt = sh['spec_exptime_pn']
            start = Time(sh['pn_date-obs']).mjd
            end = Time(sh['pn_date-end']).mjd
        if optel == 'mos':
            expt = (sh['spec_exptime_mos1'] + sh['spec_exptime_mos2']) / 2.0
            start1 = Time(sh['mos1_date-obs']).mjd
            start2 = Time(sh['mos2_date-obs']).mjd
            end1 = Time(sh['mos1_date-end']).mjd
            end2 = Time(sh['mos2_date-end']).mjd
            start = min([start1, start2])
            end = max([end1, end2])

        star = db.parse_star(specfile)
        spectbls = [utils.vecs2spectbl(w0, w1, flux, err, expt,
                                       instrument=insti, start=start, end=end,
                                       star=star, filename=specfile)]
    else:
        raise Exception('fits2tbl cannot parse data from the {} observatory.'.format(observatory))

    spec.close()

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
        err = np.zeros(N)
        expt,flags = np.zeros(N), np.zeros(N,'i1')
        start, end = [np.zeros(N)]*2
        normfac = np.ones(N)
        source = db.getinsti(specfile)*np.ones(N)
        data = [w0,w1,flux,err,expt,flags,source,normfac,start,end]
        return [__maketbl(data, specfile)]
    else:
        raise Exception('A parser for {} files has not been implemented.'.format(specfile[2:9]))

def readsav(specfile):
    """
    Reads data from IDL sav files into standardized astropy table output.
    """
    sav = spreadsav(specfile)
    wmid = sav['w140']
    flux = sav['lya_mod']
    we = mids2edges(wmid, 'left', 'linear-x')
    w0, w1 = we[:-1], we[1:]
    N = len(flux)
    err = np.zeros(N)
    expt,flags = np.zeros(N), np.zeros(N, 'i1')
    source = db.getinsti(specfile)*np.ones(N)
    normfac = np.ones(N)
    start, end = [np.zeros(N)]*2
    data = [w0,w1,flux,err,expt,flags,source,normfac,start,end]
    return [__maketbl(data, specfile)]

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
    spectbl = Table(spectbl, copy=True)

    # astropy write function doesn't store list meta correctly, so extract here
    # to add later
    sourcespecs = spectbl.meta['SOURCESPECS']
    del spectbl.meta['SOURCESPECS'] #otherwise this makes a giant nasty header
    comments = spectbl.meta['COMMENT']
    del spectbl.meta['COMMENT']

    spectbl.meta['FILENAME'] = name

    #use the native astropy function to write to fits
    spectbl.write(name, overwrite=overwrite, format='fits')

    #but open it up to do some modification
    with fits.open(name, mode='update') as ftbl:

        #add name to first table
        ftbl[1].name = 'spectrum'

        #add column descriptions
        for i,name in enumerate(spectbl.colnames):
            key = 'TDESC' + str(i+1)
            ftbl[1].header[key] = spectbl[name].description

        # add comments
        if len(comments) == 0: comments = ['']
        for comment in comments: ftbl[1].header['COMMENT'] = comment

        #add an extra bintable for the instrument identifiers
        cols = [fits.Column('instruments','13A', array=settings.instruments),
                fits.Column('bitvalues', 'I', array=settings.instvals)]
        hdr = fits.Header()
        hdr['comment'] = ('This extension is a legend for the integer '
                          'identifiers in the instrument '
                          'column of the previous extension. Instruments '
                          'are identified by bitwise flags so that they '
                          'any combination of instruments contributing to '
                          'the data wihtin a spectral element can be '
                          'identified together. For example, if instruments '
                          '4 and 16, 100 and 10000 in binary, both contribute '
                          'to the data in a bin, then that bin will have the '
                          'value 20, or 10100 in binary, to signify that '
                          'both instruments 4 and 16 have contributed. '
                          'This is identical to the handling of bitwise '
                          'data quality flags.')
        idhdu = fits.BinTableHDU.from_columns(cols, header=hdr, name='legend')
        ftbl.append(idhdu)

        #add another bintable for the sourcespecs, if needed
        if len(sourcespecs):
            maxlen = max([len(ss) for ss in sourcespecs])
            dtype = '{:d}A'.format(maxlen)
            col = [fits.Column('sourcespecs', dtype, array=sourcespecs)]
            hdr = fits.Header()
            hdr['comment'] = ('This extension contains a list of the source '
                              'files that were incorporated into this '
                              'spectrum.')
            sfhdu = fits.BinTableHDU.from_columns(col, header=hdr,
                                                  name='sourcespecs')
            ftbl.append(sfhdu)

        ftbl.flush()

def phxurl(Teff, logg=4.5, FeH=0.0, aM=0.0, repo='ftp'):
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

def phxdata(Teff, logg=4.5, FeH=0.0, aM=0.0, repo='ftp', ftpbackup=True):
    """
    Get a phoenix spectral data from the repo and return as an array.

    If ftpbackup is True, the ftp repository will be quieried if the file
    isn't found in the specified repo location and the file will be saved
    in the specified location.
    """
    path = phxurl(Teff, logg, FeH, aM, repo=repo)
    try:
        fspec = fits.open(path)
    except IOError:
        if ftpbackup and repo != 'ftp':
            warn('PHX file not found in specified repo, pulling from ftp.')
            db.fetchphxfile(Teff, logg, FeH, aM, repo=repo)
            fspec = fits.open(path)
        else:
            raise IOError('File not found at {}.'.format(path))
    return fspec[0].data

def __maketbl(data, specfile, sourcespecs=[]):
    star = specfile.split('_')[4]
    return utils.list2spectbl(data, star, specfile, sourcespecs=sourcespecs)

def __trimHSTtbl(spectbl):
    """trim off-detector portions on either end of spectbl"""
    name = path.basename(spectbl.meta['FILENAME'])
    if '_cos_' in name:
        bad = (spectbl['flags'] & 128) > 0
    elif '_sts_' in name:
        bad = (spectbl['flags'] & (128 | 4)) > 0
    beg,end = block_edges(bad)
    if len(beg) >= 2:
        return spectbl[end[0]:beg[-1]]
    elif len(beg) == 1:
        return spectbl[~bad]
    else:
        return spectbl

def writeMAST(spectbl, name, overwrite=False):
    """
    Writes spectbls to a standardized MUSCLES FITS file format that also
    includes all the keywords required for the archive.

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

    writefits(spectbl, name, overwrite=overwrite)

    with fits.open(name) as hdus:
        h0, h1, h2 = hdus