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

version = '0.0'

# new mac
gdrive = '/Users/rolo7566/Google Drive'
codepath = gdrive + '/Python/muscles'
root = gdrive + '/Grad School/Phd Work/MUSCLES'
local = '/Users/rolo7566/Datasets/MUSCLES'
datapath = local + '/data'
productspath = local + '/products'
hlsppath = productspath + '/hlsp'
scratchpath = root + '/scratchwork'
solarpath = local + '/solar'
photondir = datapath + '/photons'
flaredir = productspath + '/flare_catalogs'
proppath = root + '/share/starprops'
moviepath = productspath + '/movies'
filterpath = '/Users/rolo7566/Datasets/shared/filter response curves'

starprops = sc.SciCatalog(proppath, readOnly=True, silent=True)

datafolders = ['x-ray', 'uv', 'visible', 'ir']
bandmap = {'u':'uv', 'x':'x-ray', 'v':'visible', 'r':'ir'}

stdbandpath = root + '/settings/stdbands.json'


# old PC
#datapath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Data'
#productspath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Products'
#codepath = r'C:\Users\Parke\Google Drive\Python\muscles'
#root = r'C:\Users\Parke\Google Drive\Grad School\PhD Work\MUSCLES'


stars = starprops.indices()
observed = [star for star in stars if starprops['observed'][star]]


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

# for use in making FITS headers
HLSPtelescopes = {'hst':'HST', 'cxo':'CXO', 'xmm':'XMM', 'mod':'MOD'}
HLSPinstruments = {'cos':'COS', 'sts':'STIS', 'euv':'EUV-ESTIMATE', 'lya':'LYA-RECONSTRUCTION', 'phx':'PHX',
                   'mos':'MOS', 'pn-':'PN', 'gap':'POLYNOMIAL-FIT'}
HLSPgratings = {'g130m':'G130M', 'g160m':'G160M', 'g430l':'G430L', 'g430m':'G430M', 'g140m':'G140M', 'e230m':'E230M',
                'e230h':'E230H', 'g230l':'G230L', 'e140m':'E140M', 'fill-':'NA', '-----':'NA', 'young':'NA'}


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
    star = fos.split('_')[4]
    sets = loadsettings(star)
    pn = prenormed if sets.prenormed == [] else sets.prenormed
    isprenormed = [(s in fos) for s in pn]
    return any(isprenormed)


def settingspath(star):
    """The path for the settings file for a star."""
    return os.path.join(root, 'settings', star + '.json')


class StarSettings:
    def __init__(self, star):
        self.star = star
        self.custom_extractions = []
        self.reject_specs = []
        self.notes = []
        self.custom_ranges = {'configs':[], 'ranges':[]}
        self.norm_ranges = {'configs':[], 'ranges':[]}
        self.norm_order = []
        self.prenormed = []
        self.weird_norm = {}

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

def loadsettings(star):
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
    ss.norm_order = safeget('norm_order')
    ss.prenormed = safeget('prenormed')
    ss.weird_norm = safeget('weird_norm')
    return ss