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
from astropy.table import Table
from astropy.time import Time
from warnings import warn

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
    tbl = Table.read(tblfile[0])

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

    return Table.read(tblfile[0]), bands


def readpans(star):
    """
    Read in all panspectra for a star and return as a list.
    """
    panfiles = db.allpans(star)
    return sum(map(read, panfiles), [])

def read_panspec_sources(star):
    sets = rc.loadsettings(star)
    files, lyafile = db.panfiles(star)

    specs = read(files)

    # if there is a custom normalization order, reorder the spectra accordingly
    def insti(spec):
        inst = spec['instrument']
        assert np.all(inst == inst[0])
        return inst[0]
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
                'csv' : readcsv, 'idlsav' : readsav}
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

    if 'phx' in specfile:
        spectbl['flux'].unit = ''

    return spectbl

def readfits(specfile):
    """Read a fits file into standardized table."""

    observatory = db.parse_observatory(specfile)

    spec = fits.open(specfile)
    if any([s in specfile for s in ['coadd', 'custom', 'mod', 'panspec']]):
        return [readstdfits(specfile)]
    elif observatory == 'hst':
        sd, sh = spec[1].data, spec[1].header
        flux, err = sd['flux'], sd['error']
        shape = flux.shape
        insti = db.getinsti(specfile)
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
        if optel == 'pn---':
            expt = sh['spec_exptime_pn'] * 1000.0
            start = Time(sh['pn_date-obs']).mjd
            end = Time(sh['pn_date-end']).mjd
        if optel == 'multi':
            expt = (sh['spec_exptime_mos1'] + sh['spec_exptime_mos2'] + sh['spec_exptime_pn']) / 3.0 * 1000.0
            start1 = Time(sh['mos1_date-obs']).mjd
            start2 = Time(sh['mos2_date-obs']).mjd
            start3 = Time(sh['pn_date-obs']).mjd
            end1 = Time(sh['mos1_date-end']).mjd
            end2 = Time(sh['mos2_date-end']).mjd
            end3 = Time(sh['pn_date-end']).mjd
            start = min([start1, start2, start3])
            end = max([end1, end2, end3])
        flux, err = [spec[1].data[s] for s in ['CFlux', 'CFlux_err']]
        w0, w1 = groomwave(1)
        insti = db.getinsti(specfile)
        obsspec = utils.vecs2spectbl(w0, w1, flux, err, expt, instrument=insti, start=start, end=end, star=star,
                                     filename=specfile)

        flux = spec[2].data['Flux']
        expt, err = 0.0, 0.0
        w0, w1 = groomwave(2)
        name_pieces = obsspec.meta['NAME'].split('_')
        configuration = 'mod_apc_-----'
        name = '_'.join(name_pieces[:1] + [configuration] + name_pieces[4:])
        insti = rc.getinsti(configuration)
        modspec = utils.vecs2spectbl(w0, w1, flux, err, expt, instrument=insti, name=name, filename=specfile)

        spectbls = [obsspec, modspec]
    else:
        raise Exception('fits2tbl cannot parse data from the {} observatory.'.format(observatory))

    spec.close()

    return spectbls

def readtxt(specfile):
    """
    Reads data from text files into standardized astropy table output.
    """

    if 'young' in specfile.lower():
        data = np.loadtxt(specfile)
        if data.shape[1] == 3:
            wmid, f, _ = data.T
        elif data.shape[1] == 2:
            wmid, f = data.T
        else:
            raise ValueError('crap.')
        e = 0.0
        we = mids2edges(wmid)
        w0, w1 = we[:-1], we[1:]

        # deal with overbinning
        f_uniq = np.unique(f)
        w0_, w1_ = [], []
        for ff in f_uniq:
            keep = (ff == f)
            w0_.append(np.min(w0[keep]))
            w1_.append(np.max(w1[keep]))
        w0, w1 = map(np.array, [w0_, w1_])
        isort = np.argsort(w0)
        w0, w1, f = w0[isort], w1[isort], f_uniq[isort]

        inst = db.getinsti(specfile)
        spectbl = utils.vecs2spectbl(w0, w1, f, e, instrument=inst, filename=specfile)
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
        writehlsp(pan, overwrite=overwrite)
        dpan = read(filter(lambda s: 'dR=' in s, pfs))[0]
        writehlsp(dpan, components=False, overwrite=overwrite)
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
            band = 'v' if '430' in name else 'u'
            spectype = 'x1d' if band == 'u' else 'sx1'
            if 'gj1214' in name and 'sts_g230l' in name: spectype = 'x2d'
            f = db.findfiles(band, name, fullpaths=True)[0]
            if 'custom' in name or 'coadd' in name:
                srcspecs = fits.getdata(f, 'sourcespecs')
                srcids = [db.parse_id(s) for s in srcspecs['sourcespecs']]
                srcpaths = [db.findfiles(band, id, spectype, fullpaths=True)[0] for id in srcids]
                apertures = [fits.getval(sf, 'APERTURE') for sf in srcpaths]
                assert len(set(apertures)) == 1
                prihdr['APERTURE'] = apertures[0]
            else:
                prihdr['APERTURE'] = fits.getval(f, 'APERTURE')
        if 'xmm' in name:
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
            else:
                prihdr['DETECTOR'] = 'PN'
                prihdr['FILTER'] = hdr['pn_filter']

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
        date0 = Time(mjd0, format='mjd')
        date1 = Time(mjd1, format='mjd')
        prihdr['DATE-OBS'] = date0.isot
        prihdr['EXPSTART'] = date0.mjd
        prihdr['EXPEND'] = date1.mjd
        expt = spectbl['exptime']
        if 'xmm' in name:
            prihdr['EXPTIME'] = expt[0]
            prihdr['EXPDEFN'] = 'MEAN'
        if not np.allclose(expt, expt[0]):
            expmed = np.median(expt)
            prihdr['EXPTIME'] = expmed
            prihdr['EXPDEFN'] = 'MEDIAN'
            prihdr['EXPMAX'] = np.max(expt)
            prihdr['EXPMIN'] = np.min(expt[expt > 1])
            prihdr['EXPMED'] = expmed
        else:
            prihdr['EXPTIME'] = expt[0]

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

    prihdu = fits.PrimaryHDU(header=prihdr)

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
    fmts = ['E']*4

    if 'mod' not in name:
        cols.extend(['error', 'exptime', 'flags', 'minobsdate', 'maxobsdate'])
        descriptions.extend(['error on the flux',
                             'cumulative exposure time for the bin',
                             'data quality flags (HST data only)',
                             'modified julian date of start of first exposure',
                             'modified julian date of end of last exposure'])
        fitsnames.extend(['ERROR', 'EXPTIME', 'DQ', 'EXPSTART', 'EXPEND'])
        fmts.extend(['E']*2 + ['I'] + ['D']*2)

    if pan:
        cols.extend(['instrument', 'normfac'])
        descriptions.extend(['bitmask identifying the source instrument(s). See "instlgnd" extension for a legend.',
                             'normalization factor applied to the source spectrum'])
        fitsnames.extend(['INSTRUMENT', 'NORMFAC'])
        fmts.extend(['J', 'E'])

    for i, desc in enumerate(descriptions):
        spechdr['TDESC' + str(i+1)] = desc

    if len(spectbl.meta['COMMENT']) > 1 and not pan:
        spechdr['COMMENT'] = spectbl.meta['COMMENT']

    datas = [spectbl[col].data for col in cols]
    units = [spectbl[col].unit.to_string() for col in cols]
    fitscols = [fits.Column(array=a, name=n, format=fmt, unit=u) for a, n, fmt, u in zip(datas, fitsnames, fmts, units)]
    spechdu = fits.BinTableHDU.from_columns(fitscols, header=spechdr)
    spechdu.name = 'SPECTRUM'
    for fname, data in zip(fitsnames, datas):
        spechdu.data[fname] = data

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
        files = [db.findfiles(band, rn, spectype, fullpaths=True)[0] for rn in rootnames]
        dataids = [fits.getval(f, 'ASN_ID') for f in files]
        custom = [('custom' in s) or ('x2d' in s) for s in specnames]
        assert all(custom) or (not any(custom))
        srchdr['CUSTOM'] = custom[0], 'spectrum extracted from x2d (bad x1d)'

        fitscols = [fits.Column(name='ROOTNAME', format='9A', array=rootnames),
                    fits.Column(name='DATASET_ID', format='9A', array=dataids)]
        srchdu = fits.BinTableHDU.from_columns(fitscols, header=srchdr)
        srchdu.name = 'SRCSPECS'

        hdus.append(srchdu)
    if 'xmm' in name:
        srchdr = fits.Header()
        srchdr['COMMENT'] = ('This extension contains a list of observation IDs (DATASET_ID used for consistency '
                             'with HST data) that can be used to '
                             'locate the '
                             'data in the XMM archives. XMM data all come from only a single observation (unlike the '
                             'HST observations), but this extension is retained in place of a keyword for consistency '
                             'with the HST files.')
        srchdr['EXTNO'] = 3
        obsid = hdr['OBS_ID']
        col = fits.Column(name='DATASET_ID', format='10A', array=[obsid])
        srchdu = fits.BinTableHDU.from_columns([col], header=srchdr)
        srchdu.name = 'SRCSPECS'
        hdus.append(srchdu)

    hdus = fits.HDUList(hdus)
    hdus.writeto(hlspname, clobber=overwrite)


