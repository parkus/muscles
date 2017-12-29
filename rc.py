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

version = '22'

# new mac
gdrive = '/Users/parke/Google Drive'
codepath = gdrive + '/Python/muscles'
root = gdrive + '/Research/MUSCLES'
local = '/Users/parke/Google Drive/Datasets/muscles'
datapath = local + '/data'
photometrypath = datapath + '/photometry'
productspath = local + '/products'
hlsppath = productspath + '/hlsp'
scratchpath = root + '/scratchwork'
solarpath = local + '/solar'
photondir = datapath + '/photons'
flaredir = productspath + '/flare_catalogs'
proppath = root + '/share/starprops'
moviepath = productspath + '/movies'
filterpath = '/Users/parke/Datasets/shared/filter response curves'
sharepath = root +'/share'
xsectionpath = local + '/xsections'
normfac_file = local + '/normfac_log.json'
photref_file = photometrypath + '/photometry_refs.json'

starprops = sc.SciCatalog(proppath, readOnly=True, silent=True)
contbandpath = sharepath + '/continuum_bands.csv'
contbands = np.loadtxt(contbandpath, delimiter=',')

datafolders = ['x-ray', 'uv', 'visible', 'ir']
bandmap = {'u':'uv', 'x':'x-ray', 'v':'visible', 'r':'ir'}

stdbandpath = root + '/settings/stdbands.json'


# old PC
#datapath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Data'
#productspath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Products'
#codepath = r'C:\Users\Parke\Google Drive\Python\muscles'
#root = r'C:\Users\Parke\Google Drive\Research\MUSCLES'


stars = list(starprops.values.sort_values('Teff_muscles').index)
observed = [star for star in stars if starprops['observed'][star]]
hosts = [star for star in stars if starprops['host'][star]]

# with open(normfac_file) as f: # FIXME: uncomment once I have datasets transfered
#     normfacs = json.load(f)

insolation = 1361000. # cgs

# -----------------------------------------------------------------------------
# STANDARD BANDS

xray = [0, 100]
euv = [100, 911]
xuv = [0, 911]
fuv = [911, 1700]
nuv = [1700, 3200]
vis = [3200, 7000]
ir = [7000, np.inf]
broadband_edges = [xray[1], euv[1], fuv[1], nuv[1], vis[1]]
broadbands = [xray, euv, fuv, nuv, vis, [ir[0], 55000]]
broadband_names = ['X-ray', 'EUV', 'FUV', 'NUV', 'Visible', 'IR']


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
# phxwave = fits.getdata(os.path.join(phxrepo, 'wavegrid_hires.fits')) # FIXME: uncomment once I have datasets transfered
# phxwave = np.hstack([[499.95], midpts(phxwave), [54999.875]])


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


def seriousdqs(path_or_insti, from_x1d_header=True):
    if type(path_or_insti) is str:
        path = path_or_insti
    else:
        path = getinststr(path_or_insti)
    if '_cos_' in path:
        if from_x1d_header:
            return fits.getval(path, 'sdqflags', 1)
        else:
            if ('g130m' in path) or ('g160m' in path):
                return 8344
            else:
                return 184
    if '_sts_' in path:
        return (1 | 2 | 4 | 128 | 256 | 512 | 4096 | 8192)
    if '_fos_' in path:
        return (800 | 700 | 400 | 300 | 200 | 100 | 16)
    raise NotImplementedError('No serious dq flags defined for config\n{}'.format(path))

spectbl_format =  {'units' : ['Angstrom']*2 + ['erg/s/cm2/Angstrom']*2 + ['s','','','','d','d'],
                   'dtypes' : ['f8']*5 + ['i2', 'i4'] + ['f8']*3,
                   'fmts' : ['.2f']*2 + ['.2e']*2 + ['.1f', 'b', 'd', '.2f', '.2f', '.2f'],
                   'descriptions' : ['left (short,blue) edge of the wavelength bin',
                                     'right (long,red) edge of the wavelength bin',
                                     'average flux over the bin',
                                     'error on the flux',
                                     'cumulative exposure time for the bin',
                                     'data quality flags (specific to the instrument)',
                                     'identifier for the instrument that is the source of the data. use '
                                     'muscles.instruments[identifier] to determine the instrument',
                                     'noramlization factor applied to the source spectrum',
                                     'modified julian date of start of first exposure',
                                     'modified julian date of end of last exposure'],
                   'colnames' : ['w0','w1','flux','error','exptime','flags',
                                 'instrument','normfac','minobsdate','maxobsdate']}

stdbands = read_json(stdbandpath)

# prenormed = ['mod_lya', 'mod_euv', 'cos_g130m', 'cos_g160m', 'sts_g430l', 'sts_g430m', 'mod_apc']
prenormed = ['mod_lya', 'mod_euv', 'cos_g130m', 'cos_g160m', 'cos_g230l', 'mod_phx', 'mod_apc']
normranges = {'hst_sts_g430l':[3500., 5700.]}

lyacut = (np.array([-400, 400])/3e5 + 1)*1215.67 #[1209.67,1221.67]
panres = 1.0
norm2phot_outlier_cut = 0.01
teff_system_err = 100
gap_fit_order = 2
gap_fit_span = 20.

flare_bands = {#'hst_cos_g130m' : [[1169.5 , 1198.5 ], [1201.7 , 1212.17], [1219.17, 1271.19], [1324.83, 1426.08]],
               #'hst_cos_g130m' : [[1324.83, 1426.08]],
               'hst_cos_g130m' : [[1324.83, 1426.08]],
               'hst_cos_g160m' : [[1422.92, 1563.85], [1614.02, 1754.01]]}

specstrings = ['x1d', 'mod_euv', 'mod_lya', 'spec', 'sx1', 'mod_phx', 'coadd', 'c1f']

#listed in normalization order
instruments = ['hst_cos_g130m','hst_cos_g160m','hst_cos_g230l','hst_sts_g140m','hst_sts_e140m','hst_sts_e230m',
               'hst_sts_e230h','hst_sts_g230l','hst_sts_g430l','hst_sts_g430m','mod_gap_fill-',
               'xmm_epc_multi','xmm_epc_pn---', 'cxo_acs_-----', 'mod_euv_young', 'mod_apc_-----',
               'mod_lya_young','mod_phx_-----', 'oth_---_other', 'hst_sts_g230lb', 'hst_sts_g750l', 'hst_fos_g570h',
               'hst_fos_g780h', 'hst_cos_g140l']
instvals = [2**i for i in range(len(instruments))]
default_order = ['hst_cos_g130m','hst_cos_g160m','hst_cos_g230l','hst_sts_g140m','hst_sts_e140m','hst_sts_e230m',
                 'hst_sts_e230h','hst_sts_g230l', 'hst_sts_g230lb', 'xmm_epc_multi','xmm_epc_pn---', 'cxo_acs_-----',
                 'mod_euv_young', 'mod_apc_-----', 'mod_lya_young', 'mod_phx_-----', 'hst_sts_g750l', 'hst_sts_g430l',
                 'hst_sts_g430m', 'mod_gap_fill-', 'oth_---_other']

# for use in making plots
instranges = {'xobs': [5., 60.], 'xmm': [5., 60.], 'cxo': [1.0, 2.0], 'euv': [100., 1170.], 'hst': [1170., 5700.],
              'apec': [60., 100.], 'phx': [5700., 55000.], 'lya': lyacut, 'c130m': [1170., 1460.],
              'c160m': [1390., 1785.], 'c230l': [1670., 3190.], 's140m': [1170., 1710.], 's230m': [1605., 3070.],
              's230h': [2380., 2890.], 's230l': [1570., 3190.], 's430m': [3795., 4080.], 's430l': [2895., 5710.]}
instnames = {'xobs': 'XMM or Chandra', 'xmm': 'XMM', 'cxo': 'Chandra', 'euv': 'Empirical EUV Estimate',
             'hst': 'HST', 'apec': 'APEC Model Corona', 'phx': 'PHOENIX Model Photosphere',
             'lya': 'Ly$\\alpha$ Reconstruction', 'c130m': 'HST COS G130M', 'c160m': 'HST COS G160M',
             'c230l': 'HST COS G230L', 's140m': 'HST STIS E140M', 's230m': 'HST STIS E230M',
             's230h': 'HST STIS E230H', 's230l': 'HST STIS G230L', 's430m': 'HST STIS G430M',
             's430l': 'HST STIS G430L', 'acs':'ACIS'}
instabbrevs = {'xobs':'XMM or Chandra', 'apec':'APEC', 'euv':'Empirical EUV Estimate', 'hst':'HST', 'phx':'PHOENIX'}

# for use in making FITS headers
HLSPtelescopes = {'hst':'HST', 'cxo':'CXO', 'xmm':'XMM', 'mod':'MODEL', 'oth':'OTHER'}
HLSPinstruments = {'cos':'COS', 'sts':'STIS', 'euv':'EUV-SCALING', 'lya':'LYA-RECONSTRUCTION', 'phx':'PHX',
                   'epc':'EPIC', 'gap':'POLYNOMIAL-FIT', 'apc':'APEC', '---':'NA', 'acs':'ACIS', 'fos':'FOS'}
HLSPgratings = {'g130m':'G130M', 'g160m':'G160M', 'g430l':'G430L', 'g430m':'G430M', 'g140m':'G140M', 'e230m':'E230M',
                'e230h':'E230H', 'g230l':'G230L', 'e140m':'E140M', 'fill-':'NA', '-----':'NA', 'young':'NA',
                'pn---':'PN', 'multi':'MULTI', 'other':'NA', 'g750l':'G750L', 'g230lb':'G230LB', 'g570h':'g570H',
                'g780h':'G780H'}


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


def expand_inst_abbrv(abbrv):
    insts = filter(lambda s: abbrv in s, instruments)
    if len(insts) > 1:
        raise ValueError('More than one match.')
    elif len(insts) == 0:
        raise ValueError('No matches.')
    else:
        return insts[0]


def MASTlabels(name):
    """
    Return the MAST telescop and instrume values given a filename.
    """
    pass

foldersbyband = {'u':'uv', 'v':'visible', 'r':'ir', 'x':'x-ray'}

def dontnormalize(filename_or_spectbl):
    fos = filename_or_spectbl
    if type(fos) is not str:
        fos = fos.meta['NAME']
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
        self.tag_extractions = {}
        self.reject_specs = []
        self.notes = []
        self.custom_ranges = {'configs':[], 'ranges':[]}
        self.norm_ranges = {'configs':[], 'ranges':[]}
        self.wave_offsets = {'configs':[], 'offsets':[]}
        self.norm_order = []
        self.prenormed = []
        self.weird_norm = {}
        self.order = []

    def add_wave_offset(self, config, offset):
        self.wave_offsets['configs'].append(config)
        self.wave_offsets['offsets'].append(offset)

    def add_tag_extraction(self, config, **kwds):
        """Add a custom extraction for the tags as a config string and then kwds to provide
        to the custom extraction function in reduce."""
        config = expand_inst_abbrv(config)
        self.tag_extractions[config] = kwds

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

    def get_wave_offset(self, config):
        configmatch = lambda s: s in config
        configs = filter(configmatch, self.wave_offsets['configs'])
        if len(configs) > 1:
            raise ValueError('multiple wave offset matches')
        elif len(configs) == 1:
            i = self.wave_offsets['configs'].index(configs[0])
            return self.wave_offsets['offsets'][i]
        else:
            return None

    def get_custom_extraction(self, config):
        configmatch = lambda s: config in s['config']
        configs = filter(configmatch, self.custom_extractions)
        if len(configs) > 1:
            raise ValueError('multiple custom range matches')
        elif len(configs) == 1:
            return configs[0]['kwds']
        else:
            return None

    def get_tag_extraction(self, config):
        config = expand_inst_abbrv(config)
        try:
            return self.tag_extractions[config]
        except KeyError:
            return None

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
    ss.tag_extractions = safeget('tag_extractions')
    ss.notes = safeget('notes')
    ss.reject_specs = safeget('reject_specs')
    ss.custom_ranges = safeget('custom_ranges')
    ss.norm_ranges = safeget('norm_ranges')
    ss.norm_order = safeget('norm_order')
    ss.prenormed = safeget('prenormed')
    ss.weird_norm = safeget('weird_norm')
    ss.wave_offsets = safeget('wave_offsets')
    ss.order = safeget('order')
    return ss