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
                   'dtypes' : ['f8']*5 + ['i2']*2 + ['f8']*3,
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

prenormed = ['mod_lya', 'mod_euv', 'cos_g130m', 'cos_g160m']

lyacut = [1208.0, 1222.0]

specstrings = ['_x1d', 'mod_euv', 'mod_lya', 'xmm', 'sx1', 'mod_phx', 'coadd',
               'custom_spec']
#listed in normalization order
instruments = ['hst_cos_g130m','hst_cos_g160m','hst_cos_g230l','hst_sts_g230l',
               'hst_sts_e230m','hst_sts_e230h','hst_sts_e140m','hst_sts_g430l',
               'hst_sts_g430m','mod_gap_fill','mod_euv_-----','xmm_mos_-----',
               'mod_phx_-----','mod_lya_kevin']
instvals = [2**i for i in range(len(instruments))]
def getinsti(instrument):
    """Return the identifying number for instrument, where instrument is
    of the form, e.g., 'hst_cos_g130m'."""
    try:
        return instvals[instruments.index(instrument)]
    except ValueError:
        return -99

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

    def add_custom_extraction(self, config, **kwds):
        """Add a custom extraction os a config string and then kwds to provide
        to the custom extraction function in reduce."""
        d = {'config' : config, 'kwds' : kwds}
        self.custom_extractions.append(d)

    def add_custom_range(self, config, ranges):
        """Add good wavelength ranges for a spectrum as an Nx2 list."""
        self.custom_ranges['configs'].append(config)
        self.custom_ranges['ranges'].append(ranges)

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
    return ss