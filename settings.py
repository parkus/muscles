# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:08:18 2015

@author: Parke
"""
import json
import database as db
from numpy import reshape
from astropy.io import fits

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
            return reshape(self.custom_ranges['ranges'][i], [-1, 2])
        else:
            return None

    def get_norm_range(self, config):
        configmatch = lambda s: s in config
        configs = filter(configmatch, self.norm_ranges['configs'])
        if len(configs) > 1:
            raise ValueError('multiple custom range matches')
        elif len(configs) == 1:
            i = self.norm_ranges['configs'].index(configs[0])
            return reshape(self.norm_ranges['ranges'][i], [-1, 2])
        else:
            return None

    def add_reject(self, config, i=0):
        """Add a spectrum to the reject list as a config string and number
        specifying the segment/order to reject."""
        self.reject_specs.append([config, i])

    def save(self):
        path = db.settingspath(self.star)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

def load(star):
    path = db.settingspath(star)
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