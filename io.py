# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:28:07 2014

@author: Parke
"""

from os import path
from astropy.io import fits
import numpy as np
from mypy.my_numpy import mids2edges, block_edges, midpts
from scipy.io import readsav as spreadsav
import rc, utils, db
import astropy.table as table
from astropy.time import Time
import astropy.units as u
from warnings import warn
import os

legendcomment = ('This extension is a legend for the integer identifiers in the instrument column of the '
                  'spectrum extension. Instruments are identified by bitwise flags so that any combination of '
                  'instruments contributing to the data within a spectral element can be identified together. '
                  'For example, if instruments 4 and 16 (100 and 10000 in binary) both contribute to the data '
                  'in a bin, then that bin will have the value 20, or 10100 in binary, to signify that both '
                  'instruments 4 and 16 have contributed. This is identical to the handling of bitwise data '
                  'quality flags.')

def readphotons(star, inst):
    pf = db.findfiles('photons', star, inst, fullpaths=True)
    assert len(pf) == 1
    ph = fits.open(pf[0])
    return ph, ph['events'].data


def readFlareTbl(star, inst, label):
    tblfile = db.findfiles(rc.flaredir, star, inst, label, 'flares', fullpaths=True)
    assert len(tblfile) == 1
    tbl = table.Table.read(tblfile[0])

    w0, w1 = [], []
    i = 0
    while True:
        istr = str(i)
        if 'BANDBEG' + str(i) not in tbl.meta:
            break
        w0.append(tbl.meta['BANDBEG' + istr])
        w1.append(tbl.meta['BANDEND' + istr])
        i += 1
    bands = np.array(zip(w0, w1))

    return table.Table.read(tblfile[0]), bands


def readpans(star):
    """
    Read in all panspectra for a star and return as a list.
    """
    panfiles = db.allpans(star)
    return sum(map(read, panfiles), [])


def readpan(star):
    if star == 'sun':
        return read_solar()
    return read(db.panpath(star))[0]


def read_panspec_sources(star):
    sets = rc.loadsettings(star)
    files, lyafile = db.panfiles(star)

    specs = read(files)

    # if there is a custom normalization order, reorder the spectra accordingly
    def insti(spec):
        return rc.getinsti(db.parse_instrument(spec.meta['NAME']))
    if len(sets.norm_order) > 0:
        def key(spec):
            try:
                return sets.norm_order.index(rc.getinststr(insti(spec)))
            except ValueError:
                return insti(spec)
    # else be sure that the spectra are in the default order (the xmm spectra confuse this bc an obs and model are
    # read in)
    else:
        key = insti
    specs = sorted(specs, key=key)

    lyaspec = None if lyafile is None else read(lyafile)[0]

    return specs, lyaspec


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

    readfunc = {'fits' : readfits, 'txt' : readtxt, 'sav' : readsav,
                'csv' : readcsv, 'idlsav' : readsav, 'fit': readfits}
    star = db.parse_star(specfiles)
    i = specfiles[::-1].find('.')
    fmt = specfiles[-i:]
    specs = readfunc[fmt](specfiles)
    try:
        sets = rc.loadsettings(star)
        if 'coadd' not in specfiles and 'custom' not in specfiles:
            for config, i in sets.reject_specs:
                if config in specfiles:
                    specs.pop(i)
    except IOError:
        pass

    return specs


def readstdfits(specfile):
    """Read a fits file that was created by writefits."""
    spectbl = table.Table.read(specfile, hdu=1)
    spectbl.meta['FILENAME'] = specfile
    spectbl.meta['NAME'] = db.parse_name(specfile)
    try:
        sourcespecs = fits.getdata(specfile, 'sourcespecs')['sourcespecs']
        spectbl.meta['SOURCESPECS'] = sourcespecs
    except KeyError:
        spectbl.meta['SOURCESPECS'] = []

    if 'hst' in specfile and 'hlsp' not in specfile:
        spectbl = __trimHSTtbl(spectbl)

    if 'hlsp' in specfile:
        for col in spectbl.colnames:
            spectbl.rename_column(col, col.lower())
        spectbl.rename_column('wavelength0', 'w0')
        spectbl.rename_column('wavelength1', 'w1')
        spectbl.rename_column('dq', 'flags')
        spectbl.rename_column('expstart', 'minobsdate')
        spectbl.rename_column('expend', 'maxobsdate')
        spectbl.meta['STAR'] = db.parse_star(specfile)
        spectbl.meta['COMMENT'] = ''
    else:
        spectbl = utils.conform_spectbl(spectbl)

    if 'phx' in specfile:
        spectbl['flux'].unit = ''

    spectbl['w'] = (spectbl['w0'] + spectbl['w1'])/2

    return spectbl


def readfits(specfile, observatory=None, spectrograph=None):
    """Read a fits file into standardized table."""

    if observatory is None: observatory = db.parse_observatory(specfile)

    spec = fits.open(specfile)
    if any([s in specfile for s in ['coadd', 'custom', 'mod', 'panspec', 'other', 'hlsp']]):
        return [readstdfits(specfile)]
    elif observatory == 'hst':
        if spectrograph is None: spectrograph = db.parse_spectrograph(specfile)
        if spectrograph in ['sts', 'cos']:
            sd, sh = spec[1].data, spec[1].header
            flux, err = sd['flux'], sd['error']
            wmid, flags = sd['wavelength'], sd['dq']
            exptarr, start, end = [np.ones(flux.shape)*sh[s] for s in ['exptime', 'expstart', 'expend']]
        elif spectrograph == 'fos':
            template = specfile[:-8] + '{}' + '.fits'
            hdus = [fits.open(template.format(ext)) for ext in ['c0f', 'c1f', 'c2f', 'cqf']]
            wmid, flux, err, flags = [h[0].data[:, ::-1] for h in hdus]
            start, end = [np.ones(flux.shape)*hdus[0][0].header[s] for s in [ 'expstart', 'expend']]
            exptarr = np.ones(flux.shape) * hdus[0][1].data['exposure'][:, np.newaxis]
        else:
            raise NotImplementedError()
        shape = flux.shape
        insti = db.getinsti(specfile)
        iarr = np.ones(shape)*insti
        wedges = np.array([mids2edges(wm, 'left', 'linear-x') for wm in wmid])
        w0, w1 = wedges[:,:-1], wedges[:,1:]
        normfac = np.ones(shape)
        datas = np.array([w0,w1,flux,err,exptarr,flags,iarr,normfac,start,end])
        datas = datas.swapaxes(0,1)
        spectbls = [__maketbl(d, specfile) for d in datas]

        #cull off-detector data
        spectbls = [__trimHSTtbl(spectbl) for spectbl in spectbls]
    elif observatory == 'fuse':
        spectbls = []
        star = spec[0].header['targname']
        expt = spec[0].header['obstime']
        for sub in spec[1:]:
            w, f, e = [sub.data[s] for s in ['wave', 'flux', 'error']]
            wedges = midpts(w)
            wedges = np.insert(wedges, [0, len(wedges)], [2*w[0] - wedges[0], 2*w[-1] - wedges[-1]])
            w0, w1 = wedges[:-1], wedges[1:]
            spectbl = utils.vecs2spectbl(w0, w1, f, e, exptime=expt, star=star, filename=specfile)
            spectbls.append(spectbl)
    elif observatory in ['xmm', 'cxo']:
        sh = spec[0].header
        star = db.parse_star(specfile)

        def groomwave(ext):
            wmid = spec[ext].data['Wave']
            halfwidth = spec[ext].data['bin_width']
            w0, w1 = wmid - halfwidth, wmid + halfwidth
            assert np.allclose(w0[1:], w1[:-1])
            # there may still be slight mismatches, so fix it up
            betweens = (w0[1:] + w1[:-1]) / 2.0
            w0[1:], w1[:-1] = betweens, betweens
            return w0, w1

        # first the observed spectrum
        optel = db.parse_grating(specfile)
        if 'gj551' in specfile:
            expt = sh['pn_duration']
            start = np.nan
            end = np.nan
        elif optel == 'pn---':
            expt = sh['spec_exptime_pn'] * 1000.0
            start = Time(sh['pn_date-obs']).mjd
            end = Time(sh['pn_date-end']).mjd
        elif optel == 'multi':
            expt = (sh['spec_exptime_mos1'] + sh['spec_exptime_mos2'] + sh['spec_exptime_pn']) / 3.0 * 1000.0
            start1 = Time(sh['mos1_date-obs']).mjd
            start2 = Time(sh['mos2_date-obs']).mjd
            start3 = Time(sh['pn_date-obs']).mjd
            end1 = Time(sh['mos1_date-end']).mjd
            end2 = Time(sh['mos2_date-end']).mjd
            end3 = Time(sh['pn_date-end']).mjd
            start = min([start1, start2, start3])
            end = max([end1, end2, end3])
        elif sh['instrume'] == 'ACIS':
            starts, ends, expts = [_parse_keys_sequential(sh, root) for root in ['DATE_OBS', 'DATE_END', 'EXPTIME']]
            expt = sum(expts) * 1000.0
            starts, ends = map(Time, [starts, ends])
            start = min(starts.mjd)
            end = max(ends.mjd)
        else:
            start = 0.0
            end = 0.0
            expt = 0.0

        if '1214' not in sh['target']:
            flux, err = [spec['Obs Spectrum'].data[s] for s in ['CFlux', 'CFlux_err']]
            w0, w1 = groomwave('Obs Spectrum')
            insti = db.getinsti(specfile)
            obsspec = utils.vecs2spectbl(w0, w1, flux, err, expt, instrument=insti, start=start, end=end, star=star,
                                         filename=specfile)
            good = np.isfinite(obsspec['flux'])
            obsspec = obsspec[good]
            spectbls = [obsspec]
        else:
            spectbls = []

        # next the model
        flux = spec['Model Spectrum'].data['Flux']
        expt, err = 0.0, 0.0
        w0, w1 = groomwave('Model Spectrum')
        name_pieces = db.parse_name(specfile).split('_')
        configuration = 'mod_apc_-----'
        name = '_'.join(name_pieces[:1] + [configuration] + name_pieces[4:])
        insti = rc.getinsti(configuration)
        modspec = utils.vecs2spectbl(w0, w1, flux, err, expt, instrument=insti, name=name, filename=specfile)
        spectbls.append(modspec)
    else:
        raise Exception('fits2tbl cannot parse data from the {} observatory.'.format(observatory))

    spec.close()

    for tbl in spectbls:
        tbl['w'] = (tbl['w0'] + tbl['w1'])/2.

    return spectbls

def readtxt(specfile):
    """
    Reads data from text files into standardized astropy table output.
    """

    if 'youn' in specfile.lower():
        data = np.loadtxt(specfile)
        if data.shape[1] == 3:
            wmid, f, _ = data.T
        elif data.shape[1] == 2:
            wmid, f = data.T
        else:
            raise ValueError('crap.')
        e = 0.0
        if 'euv' in specfile:
            we = np.array([100., 200., 300., 400., 500., 600., 700., 800., 912., 1170.])
        else:
            we = mids2edges(wmid)
        w0, w1 = we[:-1], we[1:]
        f = np.interp((w0 + w1) / 2.0, wmid, f)

        inst = db.getinsti(specfile)
        spectbl = utils.vecs2spectbl(w0, w1, f, e, instrument=inst, filename=specfile)
        if 'euv' in specfile:
            spectbl = utils.evenbin(spectbl, 1.0)
        return [spectbl]
    else:
        raise Exception('A parser for {} files has not been implemented.'.format(specfile[2:9]))

def readcsv(specfile):
    if db.parse_observatory(specfile) in ['tmd', 'src']:
        data = np.loadtxt(specfile, skiprows=1, delimiter=',')
        wmid, f = data[:,1], data[:,2]
        f *= 100.0 # convert W/m2/nm to erg/s/cm2/AA
        wmid *= 10.0 # convert nm to AA
        we = np.zeros(len(wmid) + 1)
        we[1:-1] = midpts(wmid)
        dw0, dw1 = wmid[1] - wmid[0], wmid[-1] - wmid[-2]
        we[0], we[-1] = wmid[0] - dw0 / 2.0, wmid[-1] + dw1 / 2.0
        w0, w1 = we[:-1], we[1:]
        good = ~np.isnan(f)
        w0, w1, f = w0[good], w1[good], f[good]
        spectbl = utils.vecs2spectbl(w0, w1, f, filename=specfile)
        return [spectbl]
    else:
        raise Exception('A parser for {} files has not been implemented.'.format(specfile[2:9]))

def readsav(specfile):
    """
    Reads data from IDL sav files into standardized astropy table output.
    """
    sav = spreadsav(specfile)
    if 'mod_lya' in specfile:
        wmid = sav['w140']
        flux = sav['lya_mod']
    elif 'sun' in specfile:
        wmid = sav['wave'].squeeze() * 10 # nm to AA
        flux = sav['flux'].squeeze() * 100 # W m-2 nm-2
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


def get_photometry(star, lo=0.0, hi=np.inf, silent=False):
    band_dict = {'HIP:VT':'tychoV', 'HIP:BT':'tychoB', "HIP:Hp":'hipparcos',
                 'Johnson:B':'johnsonB', 'Johnson:V':'johnsonV', 'Johnson:K':'johnsonK', 'Johnson:J':'johnsonJ',
                 'Johnson:U':'johnsonU',
                 "SDSS:u'":"sdssu'", "SDSS:g'":"sdssg'", "SDSS:r'":"sdssr'", "SDSS:i'":"sdssi'", "SDSS:z'":"sdssz'",
                 "SDSS:u":"sdssu", "SDSS:g":"sdssg", "SDSS:r":"sdssr", "SDSS:i":"sdssi", "SDSS:z":"sdssz",
                 '2MASS:J':'2massJ', '2MASS:H':'2massH', '2MASS:Ks':'2massKs',
                 "Spitzer/IRAC:4.5":"spitzerI4.5", "Spitzer/IRAC:3.6":"spitzerI3.6",
                 "WISE:W1":"wiseW1", "WISE:W2":"wiseW2", "WISE:W3":"wiseW3", "WISE:W4":"wiseW4",
                 "Landolt:V":"landoltV",
                 "Cousins:I":"cousinsI", "Cousins:R":"cousinsR",
                 "VISTA:J":"vistaJ", "VISTA:H":"vistaH", "VISTA:Ks":"vistaKs",
                 'ALHAMBRA:A613M': 'alhambraA613M', 'ALHAMBRA:A457M': 'alhambraA457M',
                 'ALHAMBRA:A522M': 'alhambraA522M', 'ALHAMBRA:A646M': 'alhambraA646M',
                 'ALHAMBRA:A739M': 'alhambraA739M', 'ALHAMBRA:A921M': 'alhambraA921M',
                 'ALHAMBRA:A892M': 'alhambraA892M', 'ALHAMBRA:A708M': 'alhambraA708M',
                 'ALHAMBRA:A861M': 'alhambraA861M', 'ALHAMBRA:A425M': 'alhambraA425M',
                 'ALHAMBRA:A829M': 'alhambraA829M', 'ALHAMBRA:A394M': 'alhambraA394M',
                 'ALHAMBRA:A678M': 'alhambraA678M', 'ALHAMBRA:A491M': 'alhambraA491M',
                 'ALHAMBRA:A551M': 'alhambraA551M', 'ALHAMBRA:A802M': 'alhambraA802M',
                 'ALHAMBRA:A366M': 'alhambraA366M', 'ALHAMBRA:A770M': 'alhambraA770M',
                 'ALHAMBRA:A581M': 'alhambraA581M', 'ALHAMBRA:A948M': 'alhambraA948M',
                 'UCAC:R': 'ucacR', 'DENIS:J':'denisJ', 'DENIS:I':'denisI', 'DENIS:Ks':'denisKs'}

    tbl = table.Table.read(db.photometrypath(star))

    # select known bands
    tbl_bands = set(tbl['sed_filter'])
    known_bands = set(band_dict.keys())
    usable_bands = tbl_bands & known_bands
    if not silent:
        unusable_bands = tbl_bands - known_bands
        print "Unidentified bands: {}".format(unusable_bands)

    # load all known bands in table
    bands = {}
    for id in usable_bands:
        bandstr = band_dict[id]
        data = np.loadtxt(path.join(rc.filterpath, bandstr + '.txt'), skiprows=1)
        bands[id] = data

    # trim out of range photometry
    checkrange = lambda band: (band[0,0] > lo) and (band[-1,0] < hi)
    for id, band in bands.items():
        if not checkrange(band):
            del bands[id]
    inrange = np.array([s in bands.keys() for s in tbl['sed_filter']], bool)
    if np.sum(inrange) == 0:
        print 'No photometry covers range of {} to {} AA.'.format(lo, hi)
        return None
    tbl = tbl[inrange]

    # trim to unique photometry
    i = 0
    while i < len(tbl):
        msmt1 = tbl['sed_freq', 'sed_flux'][i]
        e1 = tbl['sed_eflux'][i]
        j = i+1
        delline = False
        while j < len(tbl):
            msmt2 =tbl['sed_freq', 'sed_flux'][j]
            e2 = tbl['sed_eflux'][j]
            if msmt1.as_void() == msmt2.as_void():
                if np.isfinite(e2):
                    if not np.isfinite(e1) or e1 == e2:
                        delline = True
                        break
                    else:
                        j += 1
                else:
                    tbl.remove_row(j)
            else:
                j += 1
        if delline == True:
            tbl.remove_row(i)
        else:
            i += 1

    return tbl, bands


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
    spectbl = table.Table(spectbl, copy=True)

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
        cols = [fits.Column('instruments','13A', array=rc.instruments),
                fits.Column('bitvalues', 'I', array=rc.instvals)]
        hdr = fits.Header()
        hdr['comment'] = legendcomment
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
            sfhdu = fits.BinTableHDU.from_columns(col, header=hdr, name='sourcespecs')
            ftbl.append(sfhdu)

        ftbl.flush()

def phxdata(Teff, logg=4.5, FeH=0.0, aM=0.0, repo='ftp', ftpbackup=True):
    """
    Get a phoenix spectral data from the repo and return as an array.

    If ftpbackup is True, the ftp repository will be quieried if the file
    isn't found in the specified repo location and the file will be saved
    in the specified location.
    """
    assert Teff in rc.phxTgrid
    assert logg in rc.phxggrid
    assert FeH in rc.phxZgrid
    assert aM in rc.phxagrid

    path = rc.phxurl(Teff, logg, FeH, aM, repo=repo)
    try:
        fspec = fits.open(path)
    except IOError:
        if ftpbackup and repo != 'ftp':
            warn('PHX file not found in specified repo, pulling from ftp.')
            rc.fetchphxfile(Teff, logg, FeH, aM, repo=repo)
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
    dqcol = 'flags' if 'flags' in spectbl.colnames else 'dq'
    if '_cos_' in name:
        bad = (spectbl[dqcol] & 128) > 0
    elif '_sts_' in name:
        bad = (spectbl[dqcol] & (128 | 4)) > 0
    elif '_fos_' in name:
        bad = np.zeros(len(spectbl), bool)
    beg,end = block_edges(bad)
    if len(beg) >= 2:
        return spectbl[end[0]:beg[-1]]
    elif len(beg) == 1:
        return spectbl[~bad]
    else:
        return spectbl

def write_simple_ascii(spectbl, name, key='flux', overwrite=False):
    """
    Write wavelength and a single spectbl column to an ascii file.
    """

    wmid = (spectbl['w0'] + spectbl['w1']) / 2.0
    f = spectbl[key]
    data = np.array([wmid, f]).T
    np.savetxt(name, data)

def writehlsp(star_or_spectbl, components=True, overwrite=False):
    """
    Writes spectbl to a standardized MUSCLES FITS file format that also
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

    if type(star_or_spectbl) is str:
        star = star_or_spectbl
        pfs = db.allpans(star)
        pan = read(filter(lambda s: 'native_resolution' in s, pfs))[0]
        writehlsp(pan, components=components, overwrite=overwrite)
        dpan = read(filter(lambda s: 'dR=' in s, pfs))[0]
        writehlsp(dpan, components=False)
        return
    else:
        spectbl = star_or_spectbl

    star = spectbl.meta['STAR']
    srcspecs = spectbl.meta['SOURCESPECS']
    name = spectbl.meta['NAME']
    pan = 'panspec' in name
    mod = 'mod' in name

    hlspname = db.hlsppath(name)

    # add a wavelength midpoint column
    w = (spectbl['w0'] + spectbl['w1']) / 2.0
    spectbl['w'] = w
    spectbl['w'].description = 'midpoint of the wavelength bin'
    spectbl['w'].unit = 'Angstrom'

    # CREATE PRIMARY EXTENSION
    prihdr = fits.Header()
    if pan:
        prihdr['TELESCOP'] = 'MULTI'
        prihdr['INSTRUME'] = 'MULTI'
        prihdr['GRATING'] = 'MULTI'

        insts = []
        for specname in srcspecs:
            tel = db.parse_observatory(specname)
            spec = db.parse_spectrograph(specname)
            grating = db.parse_info(specname, 3, 4)
            insts.append([rc.HLSPtelescopes[tel], rc.HLSPinstruments[spec], rc.HLSPgratings[grating]])
        insts = set(tuple(inst) for inst in insts)

        for i,inst in enumerate(insts):
            tkey, ikey, gkey = 'TELESC{:02d}'.format(i), 'INSTRU{:02d}'.format(i), 'GRATIN{:02d}'.format(i)
            prihdr[tkey], prihdr[ikey], prihdr[gkey] = inst
    else:
        prihdr['TELESCOP'] = rc.HLSPtelescopes[db.parse_observatory(name)]
        prihdr['INSTRUME'] = rc.HLSPinstruments[db.parse_spectrograph(name)]
        prihdr['GRATING'] = rc.HLSPgratings[db.parse_grating(name)]


        if 'hst' in name:
            # clooge. meh.
            band = name[0]
            f = db.findfiles(band, name, fullpaths=True)[0]
            aper_key = 'APER_ID' if 'fos' in name else 'APERTURE'
            if 'custom' in name or 'coadd' in name:
                srcspecs = fits.getdata(f, 'sourcespecs')
                srcids = [db.parse_id(s) for s in srcspecs['sourcespecs']]
                srcpaths = [db.sourcespecfiles(star, id)[0] for id in srcids]
                apertures = [fits.getval(sf, aper_key) for sf in srcpaths]
                assert len(set(apertures)) == 1
                prihdr['APERTURE'] = apertures[0]
            else:
                prihdr['APERTURE'] = fits.getval(f, aper_key)
        if 'xmm' in name or 'cxo' in name:
            hdr = fits.getheader(db.name2path(name))
            prihdr['GRATING'] = 'NA'
            if 'multi' in name:
                prihdr['DETECTOR'] = 'MULTI'
                prihdr['DETECT00'] = 'PN'
                prihdr['DETECT01'] = 'MOS1'
                prihdr['DETECT02'] = 'MOS2'
                prihdr['FILTER'] = 'MULTI'
                prihdr['FILTER00'] = hdr['pn_filter']
                prihdr['FILTER01'] = hdr['mos1_filter']
                prihdr['FILTER02'] = hdr['mos2_filter']
            if 'pn' in name:
                prihdr['DETECTOR'] = 'PN'
                prihdr['FILTER'] = hdr['pn_filter']
            if 'acs' in name:
                prihdr['DETECTOR'] = hdr['DETNAM']
                prihdr['FILTER'] = 'OBF'

    prihdr['TARGNAME'] = star.upper()
    prihdr['RA_TARG'] = rc.starprops['RA'][star]
    prihdr['DEC_TARG'] = rc.starprops['dec'][star]
    prihdr['PROPOSID'] = 13650
    prihdr['HLSPNAME'] = 'Measurements of the Ultraviolet Spectral Characteristics of Low-mass Exoplanet Host Stars'
    prihdr['HLSPACRN'] = 'MUSCLES'
    prihdr['HLSPLEAD'] = 'R. O. Parke Loyd'
    prihdr['PR_INV_L'] = 'France'
    prihdr['PR_INV_F'] = 'Kevin'

    if not (pan or mod):
        mjd0 = np.min(spectbl['minobsdate'])
        mjd1 = np.max(spectbl['maxobsdate'])
        if np.isfinite(mjd0):
            date0 = Time(mjd0, format='mjd')
            prihdr['DATE-OBS'] = date0.isot
            prihdr['EXPSTART'] = date0.mjd
        if np.isfinite(mjd1):
            date1 = Time(mjd1, format='mjd')
            prihdr['EXPEND'] =  date1.mjd
        expt = spectbl['exptime']
        if 'xmm' in name:
            prihdr['EXPTIME'] = expt[0]
            prihdr['EXPDEFN'] = 'MEAN'
        if 'cxo' in name:
            prihdr['EXPTIME'] = expt[0]
        if not np.allclose(expt, expt[0]):
            expmed = np.median(expt)
            prihdr['EXPTIME'] = expmed
            prihdr['EXPDEFN'] = 'MEDIAN'
            prihdr['EXPMAX'] = np.max(expt)
            prihdr['EXPMIN'] = np.min(expt[expt > 0])
            prihdr['EXPMED'] = expmed
        else:
            prihdr['EXPTIME'] = expt[0]

    if not (pan or mod) or 'phx' in name:
        try:
            inst = db.parse_instrument(name)
            normfac = rc.normfacs[star][inst][0]
            panspec = readpan(star)
            insti = rc.getinsti(inst)
            assert insti == spectbl['instrument'][0]
            normfac_vec = panspec['normfac'][panspec['normfac'] == insti]
            if len(normfac_vec) > 0:
                assert np.isclose(normfac_vec, normfac)
        except KeyError:
            normfac = 1.0
        prihdr['normfac'] = (normfac, 'normalization factor used by MUSCLES')


    prihdr['WAVEMIN'] = w[0]
    prihdr['WAVEMAX'] = w[-1]
    prihdr['WAVEUNIT'] = 'ang'
    prihdr['AIRORVAC'] = 'vac'

    if not pan or 'constant' in name:
        mid = len(w) / 2
        prihdr['SPECRES'] = w[mid]
        prihdr['WAVERES'] = w[mid+1] - w[mid]

    prihdr['FLUXMIN'] = np.min(spectbl['flux'])
    prihdr['FLUXMAX'] = np.max(spectbl['flux'])
    prihdr['FLUXUNIT'] = 'erg/s/cm2/ang' if 'phx' not in name else 'arbitrary'

    # CREATE SPECTRUM EXTENSION
    spechdr = fits.Header()
    spechdr['EXTNAME'] = 'SPECTRUM'
    spechdr['EXTNO'] = 2

    cols = ['w', 'w0', 'w1', 'flux']
    descriptions = ['midpoint of the wavelength bin',
                    'left/blue edge of the wavelength bin',
                    'right/red edge of the wavelength bin',
                    'average flux over the bin']
    fitsnames = ['WAVELENGTH', 'WAVELENGTH0', 'WAVELENGTH1', 'FLUX']
    fmts = ['D']*4

    if 'mod' not in name:
        cols.extend(['error', 'exptime', 'flags', 'minobsdate', 'maxobsdate'])
        descriptions.extend(['error on the flux',
                             'cumulative exposure time for the bin',
                             'data quality flags (HST data only)',
                             'modified julian date of start of first exposure',
                             'modified julian date of end of last exposure'])
        fitsnames.extend(['ERROR', 'EXPTIME', 'DQ', 'EXPSTART', 'EXPEND'])
        fmts.extend(['D']*2 + ['I'] + ['D']*2)

    if pan:
        # add a normalized flux column
        spectbl = utils.add_normflux(spectbl)
        spectbl['normflux'].unit = 'Angstrom-1'
        spectbl['normerr'].unit = 'Angstrom-1'
        prihdr['BOLOFLUX'] = utils.bolo_integral(spectbl.meta['STAR'])

        # add header keywords for lorentzian fit
        prihdr['LNZ_NORM'] = spectbl.meta['LNZ_NORM']
        prihdr['LNZ_GAM'] = spectbl.meta['LNZ_GAM']

        cols.extend(['instrument', 'normfac', 'normflux', 'normerr'])
        descriptions.extend(['bitmask identifying the source instrument(s). See "instlgnd" extension for a legend.',
                             'normalization factor applied to the source spectrum',
                             'flux density normalized by the bolometric flux',
                             'error on bolometrically-normalized flux density'])
        fitsnames.extend(['INSTRUMENT', 'NORMFAC', 'BOLOFLUX', 'BOLOERR'])
        fmts.extend(['J', 'D', 'D', 'D'])

    for i, desc in enumerate(descriptions):
        spechdr['TDESC' + str(i+1)] = desc

    if 'COMMENT' in spectbl.meta and len(spectbl.meta['COMMENT']) > 1 and not pan:
        spechdr['COMMENT'] = spectbl.meta['COMMENT']

    datas = [spectbl[col].data for col in cols]
    units = [spectbl[col].unit.to_string() for col in cols]
    fitscols = [fits.Column(array=a, name=n, format=fmt, unit=u) for a, n, fmt, u in zip(datas, fitsnames, fmts, units)]
    spechdu = fits.BinTableHDU.from_columns(fitscols, header=spechdr)
    spechdu.name = 'SPECTRUM'
    for fname, data in zip(fitsnames, datas):
        spechdu.data[fname] = data

    prihdu = fits.PrimaryHDU(header=prihdr)
    hdus = [prihdu, spechdu]

    # INSTRUMENT LEGEND
    if pan:
        lgndhdr = fits.Header()
        lgndhdr['comment'] = legendcomment
        lgndhdr['extno'] = 3
        lgndhdr['comment'] = ('Not all of these instruments were used to acquire data for this particular spectrum. '
                              'Therefore, not all the listed HLSP files will exist in the database. Also note that '
                              'polynomial fits for filling spectral gaps were not saved as separate spectra.')

        vals = rc.instvals
        instnames = rc.instruments
        pieces = [s.split('_') for s in instnames]
        tels, insts, gratings = zip(*pieces)
        tels = [rc.HLSPtelescopes[t] for t in tels]
        insts = [rc.HLSPinstruments[inst] for inst in insts]
        gratings = [rc.HLSPgratings[g] for g in gratings]
        dummynames = ['-_' + s + '_' + star for s in instnames]
        hlspnames = [path.basename(db.hlsppath(n)) for n in dummynames]

        names = ['BITVALUE', 'TELESCOPE', 'INSTRUMENT', 'GRATING', 'HLSP_FILE']
        datas = [vals, tels, insts, gratings, hlspnames]
        lens = [max(map(len, d)) for d in datas[1:]]
        fmts = ['J'] + [str(n) + 'A' for n in lens]
        fitscols = [fits.Column(n, fmt, array=a) for n, fmt, a in zip(names, fmts, datas)]

        lgndhdu = fits.BinTableHDU.from_columns(fitscols, header=lgndhdr)
        lgndhdu.name = 'INSTLGND'

        hdus.append(lgndhdu)

        if components:
            specs, lyaspec = read_panspec_sources(star)
            if lyaspec is not None: specs.append(lyaspec)
            for inst in instnames:
                spec = filter(lambda s: inst in s.meta['NAME'], specs)
                if len(spec) == 0:
                    continue
                assert len(spec) == 1
                spec = spec[0]
                writehlsp(spec, overwrite=overwrite)

    # SOURCE SPECTRA LIST
    if 'hst' in name:
        srchdr = fits.Header()
        srchdr['COMMENT'] = ('This extension contains a list of HST rootnames (9 character string in HST files '
                             'downloaded from MAST) and dataset IDs of the exposures used to create this spectrum '
                             'file. The dataset IDs can be used to directly locate the observations through the MAST '
                             'HST data archive search interface. Multiple identifiers indicate the spectra were '
                             'coadded.')
        srchdr['EXTNO'] = 3
        specnames = spectbl.meta['SOURCESPECS']
        if len(specnames) == 0: specnames = [name]
        rootnames = [s.split('_')[5] for s in specnames]
        files = [db.choosesourcespecs(db.findfiles(band, star, rn))[0] for rn in rootnames]
        id_key = 'ROOTNAME' if 'fos' in name else 'ASN_ID'
        dataids = [fits.getval(f, id_key) for f in files]
        custom = [('custom' in s) or ('x2d' in s) for s in specnames]
        assert all(custom) or (not any(custom))
        srchdr['CUSTOM'] = custom[0], 'spectrum extracted from x2d (bad x1d)'

        if 'gj551' in name:
            if 'g230lb' in name:
                rootnames = dataids = ['OCR7QQANQ', 'OCR7QQANQ']
            if 'g430l' in name:
                rootnames = dataids = ['OCR7QQAOQ', 'OCR7QQAPQ']
            if 'g750l' in name:
                rootnames = dataids = ['OCR7QQARQ', 'OCR7QQASQ', 'OCR7QQAQQ']

        fitscols = [fits.Column(name='ROOTNAME', format='9A', array=rootnames),
                    fits.Column(name='DATASET_ID', format='9A', array=dataids)]
        srchdu = fits.BinTableHDU.from_columns(fitscols, header=srchdr)
        srchdu.name = 'SRCSPECS'

        hdus.append(srchdu)
    if 'xmm' in name or 'cxo' in name:
        srchdr = fits.Header()
        if 'xmm' in name:
            srchdr['COMMENT'] = ('This extension contains a list of observation IDs (DATASET_ID used for consistency '
                                 'with HST data) that can be used to locate the data in the XMM archives. XMM data '
                                 'all come from only a single observation (unlike the HST observations), '
                                 'but this extension is retained in place of a keyword for consistency with the HST '
                                 'files.')
        if 'cxo' in name:
            srchdr['COMMENT'] = ('This extension contains a list of observation IDs (DATASET_ID used for consistency '
                                 'with HST data) that can be used to locate the data in the CXO archives.')
        srchdr['EXTNO'] = 3
        obsids = _parse_keys_sequential(hdr, 'OBS_ID')
        col = fits.Column(name='DATASET_ID', format='10A', array=obsids)
        srchdu = fits.BinTableHDU.from_columns([col], header=srchdr)
        srchdu.name = 'SRCSPECS'
        hdus.append(srchdu)

    hdus = fits.HDUList(hdus)
    hdus.writeto(hlspname, clobber=overwrite)


def read_xsections(species, dissoc_only=True):
    name = path.join(rc.xsectionpath, species.upper())
    with open(name) as f:
        ionlim = float(f.readline())*10.0 # Å
        data = np.loadtxt(f)
    names = ['w', 'x', 'dx/dT']
    Nbranches = (data.shape[1] - 3)/2
    for i in range(Nbranches):
        names.extend(['y_{}'.format(i), 'dy/dT_{}'.format(i)])
    tbl = table.Table(data=data, names=names)
    tbl.meta['Nbranches'] = Nbranches
    tbl.meta['ion_limit'] = ionlim

    tbl['w'] *= 10 # nm to Å

    if dissoc_only:
        keep = tbl['w'] > ionlim
        tbl = tbl[keep]

    if species == 'H2O':
        logx = np.log10(tbl['x'][-2:])
        logw = np.log10(tbl['w'][-2:])
        m = np.diff(logx) / np.diff(logw)
        dw = 0.1
        neww = np.arange(tbl['w'][-1], 2400+dw, dw)
        newx = 10**(m*(np.log10(neww) - logw[0]) + logx[0])
        newdata = np.array([neww, newx, np.zeros_like(newx), np.ones_like(newx), np.zeros_like(newx)]).T
        newtbl = table.Table(data=newdata, names=names)
        tbl = table.vstack([tbl, newtbl])

    return tbl


_si2cgs_flux = lambda x: (x * u.W / u.m**2 / u.nm).to(u.erg/u.s/u.AA/u.cm**2)


def _parse_keys_sequential(hdr, key_root):
    lst = []
    if key_root in hdr:
        lst.append(hdr[key_root])
    i = 0
    while True:
        key = key_root + str(i)
        if key in hdr:
            lst.append(hdr[key])
        else:
            if i > 0:
                return lst
        i += 1




def read_solar(period='active bright'):
    filename = path.join(rc.solarpath, 'WHI_reference_spectra.dat')
    data = np.loadtxt(filename, skiprows=142)
    w0, fluxes = data[:,0], data[:,1:4].T
    w1 = np.append(w0[1:], w0[-1] + 0.1)
    w0, w1 = 10*w0*u.AA, 10*w1*u.AA

    periods = ['active dark', 'active bright', 'quiet']
    flux = fluxes[periods.index(period)]
    flux = _si2cgs_flux(flux)

    flux_units = u.erg / u.s / u.cm**2 / u.AA
    name = 'p_multi_-_-_sun_{}_spectrum'.format(period.replace(' ', '_'))
    spectbl = utils.vecs2spectbl(w0, w1, flux, star='sun', name=name, filename=filename)
    return spectbl

