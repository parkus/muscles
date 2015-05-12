# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 15:51:54 2014

@author: Parke
"""
import os
from astropy.io import fits
from astropy.table import Table
from pandas import read_pickle
from mypy.my_numpy import midpts, inranges
import numpy as np
import io, settings
from itertools import product as iterproduct
from urllib import urlretrieve
from math import ceil


# new mac
gdrive = '/Users/rolo7566/Google Drive'
codepath = gdrive + '/Python/muscles'
root = gdrive + '/Grad School/Phd Work/MUSCLES'
local = '/Users/rolo7566/Datasets/MUSCLES'
datapath = local + '/data'
productspath = local + '/products'

target_list = root + '/target_list.txt'

bandmap = {'u':'uv', 'x':'x-ray', 'v':'visible', 'r':'ir'}

# old PC
#datapath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Data'
#productspath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Products'
#codepath = r'C:\Users\Parke\Google Drive\Python\muscles'
#root = r'C:\Users\Parke\Google Drive\Grad School\PhD Work\MUSCLES'

# -----------------------------------------------------------------------------
# PHOENIX DATABASE
phxrepo = os.path.join(datapath, 'phoenix')
phxTgrid = np.hstack([np.arange(2300,7000,100),
                   np.arange(7000,12001,200)])
phxggrid = np.arange(0.0, 6.1, 0.5)
phxZgrid = np.hstack([np.arange(-4.0, -2.0, 1.0),
                   np.arange(-2.0, 1.1, 0.5)])
phxagrid = np.arange(-0.2, 1.3, 0.2)
phxgrids = [phxTgrid, phxggrid, phxZgrid, phxagrid]
phxwave = fits.getdata(os.path.join(phxrepo, 'wavegrid_hires.fits'))
phxwave = np.hstack([[499.95], midpts(phxwave), [54999.875]])

def fetchphxfile(Teff, logg, FeH, aM, repo=phxrepo):
    loc, ftp = [io.phxurl(Teff, logg, FeH, aM, repo=r) for r in [repo, 'ftp']]
    urlretrieve(ftp, loc)

def fetchphxfiles(Trng=[2500,3500], grng=[4.0,5.5], FeHrng=[0.0, 0.0],
                  aMrng=[0.0, 0.0], repo=phxrepo):
    """
    Download all Phoenix spectra covering the provided ranges. Does not
    re-download files that already exist in the directory.

    Default values are from UV variability sample properties.
    """
    def makerng(x, grid):
        if not hasattr(x, '__iter__'):
            x = [np.max(grid[grid < x]), np.min(grid[grid > x])]
        return x
    rngs = [Trng, grng, FeHrng, aMrng]
    rngs = map(makerng, rngs, phxgrids)

    def inclusive(grid, rng):
        use = np.logical_and(grid >= rng[0], grid <= rng[1])
        return grid[use]
    grids = map(inclusive, phxgrids, rngs)

    combos = iterproduct(*grids)
    paths = []
    for combo in combos:
        locpath = io.phxurl(*combo, repo=repo)
        if not os.path.exists(locpath):
            paths.append((locpath, io.phxurl(*combo, repo='ftp')))

    N = len(paths)
    datasize = N*6.0/1024.0
    print ('Beginning download of {} files, {:.3f} Gb. Ctrl-C to stop.'
           ''.format(N,datasize))
    print 'Progress bar (- = 5%):\n'

    Nbar = ceil(N/20.0)
    for i,(loc, ftp) in enumerate(paths):
        if i % Nbar == 0:
            print '-',
        urlretrieve(ftp,loc)

def phxpath(star):
    """Standard name for interpolated phoenix spectrum."""
    name = 'r_mod_phx_-----_{}_interpolated.fits'.format(star)
    return os.path.join(datapath, 'ir', name)

# -----------------------------------------------------------------------------
# STELLAR PROPERTIES DATABASE

proppath = os.path.join(root, 'star_properties.p')
refpath = os.path.join(root, 'star_properties_references.p')
errpospath = os.path.join(root, 'star_properties_errors_pos.p')
errnegpath = os.path.join(root, 'star_properties_errors_neg.p')

props = read_pickle(proppath)
proprefs = read_pickle(refpath)
properrspos =read_pickle(errpospath)
properrsneg = read_pickle(errnegpath)

proptables = [props, proprefs, properrspos, properrsneg]
proppaths = [proppath, refpath, errpospath, errnegpath]

stars = list(props.index)

def __setormask(tbl, path, star, prop, value):
    if value is None:
        isstr = isinstance(tbl[prop][star], basestring)
        tbl[prop][star] = '' if isstr else np.nan
    else:
        tbl[prop][star] = value
    tbl.to_pickle(path)

def setprop(star, prop, value, poserr=None, negerr=None, ref=None):
    setit = lambda tbl, path, value: __setormask(tbl, path, star, prop, value)
    setit(props, proppath, value)
    setit(properrspos, errpospath, poserr)
    setit(properrsneg, errnegpath, negerr)
    setit(proprefs, refpath, ref)

def printprop(star, name):
    tbls = [props, properrsneg, properrspos, proprefs]
    x, en, ep, ref = [t[name][star] for t in tbls]
    if ref == '': ref = 'no reference'
    print '{:f} -{:f} +{:f}\t{}'.format(x, en, ep, ref)

def phxinput(star):
    Teff = props['Teff'][star]
    kwds = {}
    for key in ['logg', 'FeH', 'aM']:
        x = props[key][star]
        if ~np.isnan(x):
            kwds[key] = x
    return Teff, kwds

def props2csv(folder):
    for tbl, path in zip(proptables, proppaths):
        path = os.path.basename(path)
        csvpath = path.replace('.p', '.csv')
        csvpath = os.path.join(folder, csvpath)
        tbl.to_csv(csvpath)

# -----------------------------------------------------------------------------
# AIRGLOW LINES

airglow_path = os.path.join(root, 'airglow_lines.csv')
airglow_lines = Table.read(airglow_path)

# -----------------------------------------------------------------------------
# SPECTRAL DATA ORGANIZATION

def findfiles(path_or_band, *substrings):
    """Look for a files in directory at path that contains ALL of the strings
    in substrings in its filename."""
    if not os.path.exists(path_or_band):
        band = path_or_band if len(path_or_band) > 1 else bandmap[path_or_band]
        path_or_band = datapath + '/' + band
    def good(name):
        hasstring = [(s in name) for s in substrings]
        return all(hasstring)
    return filter(good, os.listdir(path_or_band))

def validpath(name):
    if os.path.exists(name):
        return name
    else:
        band = name[0]
        folder = bandmap[band]
        path = os.path.join(datapath, folder, name)
        if not os.path.exists(path):
            raise IOError("Can't find file {} in the standard place ({})."
                          "".format(name, os.path.join(datapath, folder)))
        else:
            return path


def findsimilar(specfile, newstring):
    """Find a file with the same identifier as sepcfile, but that also contains
    newstring in the file name. For example, find the the coadd version of the
    u_hst_cos_g130m_gj832 observation."""
    base = parse_id(specfile)
    dirname = os.path.dirname(specfile)
    names = findfiles(dirname, base, newstring)
    paths = [os.path.join(dirname, n) for n in names]
    return paths

def configfiles(star, configstring):
    """Find the spectra for the star that match configstring."""
    allfiles = allspecfiles(star)
    return filter(lambda f: configstring in f, allfiles)

def choosesourcespecs(specfiles):
    """Given a list of specfiles, select the best out of duplicated files and
    remove custom coadds and extractions."""
    #get rid of reduced files
    specfiles = filter(lambda s: not ('coadd' in s or 'custom' in s), specfiles)

    #remove any non-spec files
    specfiles = filter(isspec, specfiles)

    #for x1dsum files, get rid of any x1ds included in the x1dsum
    xsums = filter(lambda s: 'x1dsum' in s, specfiles)
    x1ds = filter(lambda s: 'x1d.fits' in s, specfiles)
    if len(xsums) > 0:
        hdrs = [fits.getheader(xs, 1) for xs in xsums]
        rngs = [[h['expstart'], h['expend']] for h in hdrs]
        rngs = sorted(rngs)
        def covered(x1d):
            start = fits.getval(x1d, 'expstart', 1)
            return inranges(start, rngs, [1,1])
        badx1ds = filter(covered, x1ds)
        for bx in badx1ds: specfiles.remove(bx)

    return specfiles

def sourcespecfiles(star, configstring):
    """Source (not coadd or custom) spectrum files that conatin configstring."""
    return choosesourcespecs(configfiles(star, configstring))

def coaddfile(star, configstring):
    """The coadd file for a config and star."""
    allfiles = allspecfiles(star)
    f = filter(lambda f: configstring in f and 'coadd' in f, allfiles)
    if len(f) > 1:
        raise Exception('Multiple files found.')
    else:
        return f[0]

def customfile(star, configstring):
    """The custom extraction file for a config and star."""
    allfiles = allspecfiles(star)
    f = filter(lambda f: configstring in f and 'custom_spec' in f, allfiles)
    if len(f) > 1:
        raise Exception('Multiple files found.')
    else:
        return f[0]

isspec = lambda name: any([s in name for s in settings.specstrings])

def allspecfiles(star):
    """Find all the spectra for the star within the subdirectories of path
    using the file naming convention."""
    hasstar = lambda name: star in name

    subfolders = [datapath]
    contents = [os.path.join(datapath,p) for p in os.listdir(datapath)]
    subfolders.extend(filter(os.path.isdir, contents))
    subfolders = filter(lambda f: 'phoenix' not in f, subfolders)
    files = []
    for sf in subfolders:
        allfiles = os.listdir(sf)
        starfiles = filter(hasstar, allfiles)
        specfiles = filter(isspec, starfiles)
        specfiles = [os.path.join(datapath, sf, f) for f in specfiles]
        files.extend(specfiles)

    return files

def allsourcefiles(star):
    """All source spectrum files for a star."""
    allfiles = allspecfiles(star)
    return choosesourcespecs(allfiles)

def coaddgroups(star, config=None, nosingles=False):
    """Return a list of groups of files that should be coadded.
    Chooses the best source files and avoids dulicates."""
    allfiles = allsourcefiles(star)
    filterfiles = lambda s: filter(lambda ss: s in ss, allfiles)
    if config:
        files = filterfiles(config)
    files = map(filterfiles, settings.instruments)
    files = filter(len, files)
    if nosingles:
        files = filter(lambda x: len(x) > 1, files)
    if config:
        files = files[0]
    return files

def panfiles(star):
    """Return the files for the spectra to be spliced into a panspectrum,
    replacing "raw" files with coadds and custom extractions as appropriate
    and ordering according to how the spectra should be normalized."""
    #FIXME: not replacing coadds properly for eps eri

    allfiles = allsourcefiles(star)
    use = lambda name: any([s in name for s in settings.instruments])
    allfiles = filter(use, allfiles)
    filterfiles = lambda s: filter(lambda ss: s in ss, allfiles)
    files = map(filterfiles, settings.instruments)
    files = reduce(lambda x,y: x+y, files)

    #sub in custom extractions
    files = sub_coaddfiles(files)
    files = sub_customfiles(files)

    #parse out lya file
    lyafile = filter(lambda f: settings.instruments[13] in f, files)
    assert len(lyafile) <= 1
    if len(lyafile):
        lyafile = lyafile[0]
        files.remove(lyafile)
    else:
        lyafile = None

    return files, lyafile

def lyafile(star):
    """Find the file with the best Lya data for star."""
    files = findfiles('uv', star, 'sts', '140')
    files = filter(isspec, files)
    files = [os.path.join(datapath, 'uv', f) for f in files]
    files = sub_customfiles(files)
    files = sub_coaddfiles(files)
    if len(files) > 1:
        raise ValueError('More than one file found:\n' + '\n\t'.join(files))
    else:
        return os.path.basename(files[0])

def parse_info(filename, start, stop):
    """Parse out the standard information bits from a muscles filename."""
    name = os.path.basename(filename)
    pieces = name.split('_')
    slc = slice(start, stop)
    return '_'.join(pieces[slc])

def parse_instrument(filename):
    return parse_info(filename, 1, 4)
def parse_spectrograph(filename):
    return parse_info(filename, 2, 3)
def parse_band(filename):
    return parse_info(filename, 0, 1)
def parse_star(filename):
    return parse_info(filename, 4, 5)
def parse_id(filename):
    return parse_info(filename, 0, 5)
def parse_observatory(filename):
    return parse_info(filename, 1, 2)
def parse_paninfo(filename):
    return parse_info(filename, 6, None)
def parse_name(filename):
    name = os.path.basename(filename)
    return name.split('.')[0]

def allpans(star):
    """All panspec files for a star."""
    allfiles = os.listdir(productspath)
    identifier = '{}_panspec'.format(star)
    panfiles = filter(lambda s: identifier in s, allfiles)
    return [os.path.join(productspath, pf) for pf in panfiles]

def panpath(star):
    """The native resolution panspec file for a star."""
    name = 'p_msl_pan_-----_{}_panspec_native_resolution.fits'.format(star)
    return os.path.join(productspath, name)

def Rpanpath(star, R):
    """The constant R panspec file for a star."""
    name = ('p_msl_pan_-----_{}_panspec_constant_R={:d}.fits'
            ''.format(star, int(round(R))))
    return os.path.join(productspath, name)

def dpanpath(star, dR):
    """The constant resolution (binsize) panspec file for a star."""
    name = ('p_msl_pan_-----_{}_panspec_constant_dR={:.1f} angstrom.fits'
            ''.format(star, float(dR)))
    return os.path.join(productspath, name)

def settingspath(star):
    """The path for the settings file for a star."""
    return os.path.join(root, 'settings', star+'.json')

def getinsti(filename):
    """Returns the numeric identifier for the instrument that created a
    spectrum based on the filename."""
    return settings.getinsti(parse_instrument(filename))

def group_by_instrument(lst):
    """Group the spectbls by instrument, returning a list of the groups. Useful
    for coaddition. Preserves order. lst can be a list of filenames or a list
    of spectbls."""

    #get the unique instruments
    if type(lst[0]) is str:
        specfiles = lst
    else:
        specfiles = [spec.meta['FILENAME'] for spec in lst]
    allinsts = np.array(map(parse_instrument, specfiles))
    insts, ind = np.unique(allinsts, return_index=True)
    insts = insts[np.argsort(ind)]

    #group em
    groups = []
    for inst in insts:
        use = np.nonzero(allinsts == inst)[0]
        specgroup = [lst[i] for i in use]
        groups.append(specgroup)

    return groups

def coaddpath(specpath):
    """Construct standardized name for coadd FITS file within same directory as
    specfile."""
    specdir = os.path.dirname(specpath)
    specname = os.path.basename(specpath)
    parts = specname.split('_')
    coaddname = '_'.join(parts[:5]) + '_coadd.fits'
    return os.path.join(specdir, coaddname)

def sub_coaddfiles(specfiles):
    """Replace any group of specfiles from the same instrument with a coadd
    file that includes data from all spectra in that group if one exists in the
    same directory.
    """
    groups = group_by_instrument(specfiles)
    result = []
    for group in groups:
        group = filter(lambda s: 'coadd' not in s, group)
        coaddfile = find_coaddfile(group)
        if coaddfile is not None:
            result.append(coaddfile)
        else:
            result.extend(group)
    return result

def sub_customfiles(specfiles):
    """Replace any file with a custom extraction file for the same instrument
    if one exists in the same directory."""
    result = []
    for name in specfiles:
        customfiles = findsimilar(name, 'custom')
        if len(customfiles) > 1:
            raise ValueError('Multiple matching files.')
        elif len(customfiles) == 1:
            if customfiles[0] not in result:
                result.append(customfiles[0])
        else:
            result.append(name)
    return result

def find_coaddfile(specfiles):
    """
    Look for a file that is the coaddition of the provided spectlbs.
    Returns the filename if it exists and it contains data from all of the
    provided spectbls, otherwise returns none.
    """
    #check for multiple configurations
    insts = np.array(map(parse_instrument, specfiles))
    if any(insts[:-1] != insts[:-1]):
        return NotImplemented("...can't deal with different data sources.")

    coaddfile = coaddpath(specfiles[0])
    if os.path.isfile(coaddfile):
        coadd, = io.read(coaddfile)

        #check that the coadd contains the same data as the spectbls
        #return none if any is missing
        csourcespecs = coadd.meta['SOURCESPECS']
        for sf in specfiles:
            if sf not in csourcespecs:
                return None
        return coaddfile

    #if the file doesn't exist, return None
    else:
        return None

def auto_rename(folder):
    """
    Rename all of the files in the folder according to the standard naming
    convention as best as possible.
    """

    #find all the FITS files
    names = filter(lambda s: s.endswith('.fits'), os.listdir(folder))

    tele = None
    unchanged = []
    while len(names) > 0:
        name = names.pop(0)
        try:
            filepath = os.path.join(folder, name)
            hdr = fits.getheader(filepath)

            telekeys = ['telescop']
            for telekey in telekeys:
                try:
                    tele = hdr[telekey]
                    break
                except:
                    tele = None

            if tele is None:
                unchanged.append(name)
                continue
            if tele == 'HST':
                #using the x1d to get the appropriate info, rename all the files
                #from the same observation
                try:
                    root = hdr['rootname']
                except KeyError:
                    root = name[-18:-9]
                def isspec(s):
                    return (root + '_x1d.fits') in s or (root + '_sx1.fits') in s

#                specfile = name if isspec(name) else filter(isspec, names)[0]
#                xpath = os.path.join(folder,specfile)
#                xhdr = fits.getheader(xpath)
                inst = hdr['instrume']
                if inst == 'STIS': inst = 'STS'
                grating = hdr['opt_elem']
                star = hdr['targname']
                cenwave = hdr['cenwave']
                band = 'U' if cenwave < 4000.0 else 'V'

                obsnames = filter(lambda s: root in s, names) + [name]
                for oname in obsnames:
                    try:
                        names.remove(oname)
                    except ValueError:
                        pass
                    opath = os.path.join(folder, oname)
                    original_name = fits.getval(opath, 'filename')
                    newname = '_'.join([band, tele, inst, grating, star,
                                        original_name])
                    os.rename(opath, os.path.join(folder, newname.lower()))
        except:
            unchanged.append(name)
            continue

    if len(unchanged) > 0:
        print 'The following files could not be renamed:'
        for name in unchanged: print '    ' + name
