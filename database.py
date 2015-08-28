# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 15:51:54 2014

@author: Parke
"""
import os
from mypy.my_numpy import midpts
import numpy as np
from itertools import product as iterproduct
from urllib import urlretrieve
from math import ceil
import json
from astropy.io import fits
from pandas import read_json
import scicatalog.scicatalog as sc


# new mac
gdrive = '/Users/rolo7566/Google Drive'
codepath = gdrive + '/Python/muscles'
root = gdrive + '/Grad School/Phd Work/MUSCLES'
local = '/Users/rolo7566/Datasets/MUSCLES'
datapath = local + '/data'
productspath = local + '/products'
scratchpath = root + '/scratchwork'
solarpath = local + '/solar'
photondir = datapath + '/photons'
flaredir = productspath + '/flare_catalogs'
proppath = root + '/share/starprops'
moviepath = productspath + '/movies'
sharepath = root + '/share'

#TODO: clean this up -- move to using the starprops
starprops = sc.SciCatalog(sharepath + '/starprops')
target_list = root + '/share/target_list.txt'
target_list_tex = root + '/share/target_list_tex.txt'
observed_list = root + '/share/observed_list.txt'

datafolders = ['x-ray', 'uv', 'visible', 'ir']
bandmap = {'u':'uv', 'x':'x-ray', 'v':'visible', 'r':'ir'}

stdbandpath = root + '/settings/stdbands.json'


# old PC
#datapath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Data'
#productspath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Products'
#codepath = r'C:\Users\Parke\Google Drive\Python\muscles'
#root = r'C:\Users\Parke\Google Drive\Grad School\PhD Work\MUSCLES'


with open(target_list) as f:
    stars = f.read().splitlines()
    stars = [s.replace('eps eri', 'v-eps-eri') for s in stars]

with open(observed_list) as f:
    observed = f.read().splitlines()

# -----------------------------------------------------------------------------
# PHOENIX DATABASE
phoenixbaseurl = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
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
        return phoenixbaseurl + folder + name
    else:
        return os.path.join(repo, name)

def fetchphxfile(Teff, logg, FeH, aM, repo=phxrepo):
    loc, ftp = [phxurl(Teff, logg, FeH, aM, repo=r) for r in [repo, 'ftp']]
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
        locpath = phxurl(*combo, repo=repo)
        if not os.path.exists(locpath):
            paths.append((locpath, phxurl(*combo, repo='ftp')))

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
# AIRGLOW LINES

airglow_path = os.path.join(root, 'airglow_ranges.csv')
airglow_ranges = np.loadtxt(airglow_path, delimiter=',')

# -----------------------------------------------------------------------------
# SPECTRAL DATA ORGANIZATION

def findfiles(path_or_band, *substrings, **kwargs):
    """Look for a files in directory at path that contains ALL of the strings
    in substrings in its filename. Add fullpaths=True if desired."""
    if not os.path.exists(path_or_band):
        band = path_or_band if len(path_or_band) > 1 else bandmap[path_or_band]
        path_or_band = datapath + '/' + band
    def good(name):
        hasstring = [(s in name) for s in substrings]
        return all(hasstring)
    files = filter(good, os.listdir(path_or_band))
    if 'fullpaths' in kwargs and kwargs['fullpaths'] == True:
        files = [os.path.join(path_or_band, f) for f in files]
    return files

def validpath(name):
    if os.path.exists(name):
        return name
    else:
        band = name[0]
        folder = bandmap[band]
        path = os.path.join(datapath, folder, name)
        if path[-4:] != 'fits':
            path += '.fits'
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
    """Given a list of specfiles, remove coadds and replace originals
    with custom files."""
    #get rid of reduced files
    specfiles = filter(lambda s: not ('coadd' in s or 'custom' in s), specfiles)

    #remove any non-spec files
    specfiles = filter(isspec, specfiles)

    return specfiles

def sourcespecfiles(star, configstring):
    """Source spectrum files that conatin configstring."""
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


isspec = lambda name: any([s in name for s in specstrings])

def allspecfiles(star):
    """Find all the spectra for the star within the subdirectories of path
    using the file naming convention."""
    hasstar = lambda name: star in name

    folders = [os.path.join(datapath,p) for p in datafolders]
    files = []
    for sf in folders:
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

def coaddgroups(star, nosingles=False):
    """Return a list of groups of files that should be coadded.
    Chooses the best source files and avoids dulicates."""
    allfiles = allsourcefiles(star)
    allfiles = sub_customfiles(allfiles)
    filterfiles = lambda s: filter(lambda ss: s in ss, allfiles)
    files = map(filterfiles, instruments)
    files = filter(len, files)
    if nosingles:
        files = filter(lambda x: len(x) > 1, files)
    return files

def panfiles(star):
    """Return the files for the spectra to be spliced into a panspectrum,
    replacing "raw" files with coadds and custom extractions as appropriate
    and ordering according to how the spectra should be normalized."""
    #FIXME: not replacing coadds properly for eps eri

    allfiles = allsourcefiles(star)
    use = lambda name: any([s in name for s in instruments])
    allfiles = filter(use, allfiles)
    filterfiles = lambda s: filter(lambda ss: s in ss, allfiles)
    files = map(filterfiles, instruments)
    files = reduce(lambda x,y: x+y, files)

    #sub in custom extractions
    files = sub_customfiles(files)
    files = sub_coaddfiles(files)

    #parse out lya file
    lyafile = filter(lambda f: 'mod_lya' in f, files)
    assert len(lyafile) <= 1
    if len(lyafile):
        lyafile = lyafile[0]
        files.remove(lyafile)
    else:
        lyafile = None

    return files, lyafile

def solarfiles(date):
    files = os.listdir(solarpath)
    files = filter(lambda s: date in s, files)
    ufile = filter(lambda s: 'u' == s[0], files)[0]
    vfile = filter(lambda s: 'v' == s[0], files)[0]
    ufile, vfile = [os.path.join(solarpath, f) for f in (ufile, vfile)]
    return ufile, vfile

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
    return parse_info(filename, 0, 6)
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
    return getinsti(parse_instrument(filename))

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


def photonpath(filepath):
    fdir, fname = os.path.split(filepath)
    rootdir, _ = os.path.split(fdir)
    root = parse_info(fname, 0, 5)
    return os.path.join(rootdir, 'photons', root + '_photons.fits')


def flarepath(star, inst, label):
    inst = filter(lambda s: inst in s, instruments)
    assert len(inst) == 1
    inst = inst[0]
    name = '_'.join([inst, star, label, 'flares'])
    return os.path.join(flaredir, name + '.fits')


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
            if parse_name(sf) not in csourcespecs:
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


# -----------------------------------------------------------------------------
# "SETTINGS"

def seriousdqs(path):
    if '_cos_' in path:
        return fits.getval(path, 'sdqflags', 1)
    if '_sts_' in path:
        return (1 | 2 | 4 | 128 | 256 | 512 | 4096 | 8192)
    else:
        raise NotImplementedError('No serious dq flags defined for config\n{}'
                                  ''.format(path))
spectbl_format =  {'units' : ['Angstrom']*2 + ['erg/s/cm2/Angstrom']*2 + ['s','','','','d','d'],
                   'dtypes' : ['f8']*5 + ['i2', 'i4'] + ['f8']*3,
                   'fmts' : ['.2f']*2 + ['.2e']*2 + ['.1f', 'b', 'd', '.2f', '.2f', '.2f'],
                   'descriptions' : ['left (short,blue) edge of the wavelength bin',
                                     'right (long,red) edge of the wavelength bin',
                                     'average flux over the bin',
                                     'error on the flux',
                                     'cumulative exposure time for the bin',
                                     'data quality flags (specific to the instrument)',
                                     'identifier for the instrument that is the source of the '
                                     'data. use muscles.instruments[identifier] to determine '
                                     'the instrument',
                                     'noramlization factor applied to the '
                                     'source spectrum',
                                     'modified julian date of start of first exposure',
                                     'modified julian date of end of last exposure'],
                   'colnames' : ['w0','w1','flux','error','exptime','flags',
                                 'instrument','normfac','minobsdate','maxobsdate']}

stdbands = read_json(stdbandpath)

prenormed = ['mod_lya', 'mod_euv', 'cos_g130m', 'cos_g160m', 'sts_g430l',
             'sts_g430m']

lyacut = [1208.0, 1222.0]

flare_bands = {#'hst_cos_g130m' : [[1169.5 , 1198.5 ], [1201.7 , 1212.17], [1219.17, 1271.19], [1324.83, 1426.08]],
               #'hst_cos_g130m' : [[1324.83, 1426.08]],
               'hst_cos_g130m' : [[1324.83, 1426.08]],
               'hst_cos_g160m' : [[1422.92, 1563.85], [1614.02, 1754.01]]}

specstrings = ['x1d', 'mod_euv', 'mod_lya', 'spec', 'sx1', 'mod_phx', 'coadd']
#listed in normalization order
instruments = ['hst_cos_g130m','hst_cos_g160m','hst_sts_g430l','hst_sts_g430m',
               'hst_sts_g140m','hst_sts_e230m','hst_sts_e230h','hst_sts_g230l',
               'hst_cos_g230l','hst_sts_e140m','mod_gap_fill-','mod_euv_-----',
               'xmm_mos_-----','xmm_pn-_-----','mod_phx_-----','mod_lya_young',
               'mod_euv_young']
instvals = [2**i for i in range(len(instruments))]

def getinststr(inst_val):
    """Return the string version of an instrument value."""
    return instruments[instvals.index(inst_val)]

def getinsti(instrument):
    """Return the identifying number for instrument, where instrument is
    of the form, e.g., 'hst_cos_g130m'."""
    try:
        return instvals[instruments.index(instrument)]
    except ValueError:
        return -99

def MASTlabels(name):
    """
    Return the MAST telescop and instrume values given a filename.
    """
    pass

foldersbyband = {'u':'uv', 'v':'visible', 'r':'ir', 'x':'x-ray'}

def dontnormalize(filename_or_spectbl):
    fos = filename_or_spectbl
    if type(fos) is not str:
        fos = fos.meta['FILENAME']
    isprenormed = [(s in fos) for s in prenormed]
    return any(isprenormed)

class StarSettings:
    def __init__(self, star):
        self.star = star
        self.custom_extractions = []
        self.reject_specs = []
        self.notes = []
        self.custom_ranges = {'configs':[], 'ranges':[]}
        self.norm_ranges = {'configs':[], 'ranges':[]}

    def add_custom_extraction(self, config, **kwds):
        """Add a custom extraction os a config string and then kwds to provide
        to the custom extraction function in reduce."""
        d = {'config' : config, 'kwds' : kwds}
        self.custom_extractions.append(d)

    def add_custom_range(self, config, ranges):
        """Add good wavelength range for a spectrum."""
        self.custom_ranges['configs'].append(config)
        self.custom_ranges['ranges'].append(ranges)

    def add_norm_range(self, config, ranges):
        """Add good wavelength range for a spectrum."""
        self.norm_ranges['configs'].append(config)
        self.norm_ranges['ranges'].append(ranges)

    def get_custom_range(self, config):
        configmatch = lambda s: s in config
        configs = filter(configmatch, self.custom_ranges['configs'])
        if len(configs) > 1:
            raise ValueError('multiple custom range matches')
        elif len(configs) == 1:
            i = self.custom_ranges['configs'].index(configs[0])
            return np.reshape(self.custom_ranges['ranges'][i], [-1, 2])
        else:
            return None

    def get_norm_range(self, config):
        configmatch = lambda s: s in config
        configs = filter(configmatch, self.norm_ranges['configs'])
        if len(configs) > 1:
            raise ValueError('multiple custom range matches')
        elif len(configs) == 1:
            i = self.norm_ranges['configs'].index(configs[0])
            return np.reshape(self.norm_ranges['ranges'][i], [-1, 2])
        else:
            return None

    def add_reject(self, config, i=0):
        """Add a spectrum to the reject list as a config string and number
        specifying the segment/order to reject."""
        self.reject_specs.append([config, i])

    def save(self):
        path = settingspath(self.star)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

def load(star):
    path = settingspath(star)
    with open(path) as f:
        d = json.load(f)
    def safeget(key):
        if key in d:
            return d[key]
        else:
            defaultobject = StarSettings('default')
            return defaultobject.__dict__[key]
    ss = StarSettings(star)
    ss.custom_extractions = safeget('custom_extractions')
    ss.notes = safeget('notes')
    ss.reject_specs = safeget('reject_specs')
    ss.custom_ranges = safeget('custom_ranges')
    ss.norm_ranges = safeget('norm_ranges')
    return ss