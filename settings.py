# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:08:18 2015

@author: Parke
"""
import json
import database as db

dqmask = {'cos' : (64 | 2048 | 1 | 256 | 128 | 32 | 16),
          'sts' : (1 | 2 | 4 | 128 | 256 | 512 | 4096 | 8192)}
          
spectbl_format =  {'units' : ['Angstrom']*2 + ['erg/s/cm2/Angstrom']*2 + ['s','',''],
                   'dtypes' : ['f8']*5 + ['i2', 'i2'],
                   'fmts' : ['.2f']*2 + ['.2e']*2 + ['.1f', 'b', 'd'],
                   'descriptions' : ['left (short,blue) edge of the wavelength bin',
                                     'right (long,red) edge of the wavelength bin',
                                     'average flux over the bin',
                                     'error on the flux',
                                     'cumulative exposure time for the bin',
                                     'data quality flags (specific to the instrument)',
                                     'identifier for the instrument that is the source of the '
                                     'data. use muscles.instruments[identifier] to determine '
                                     'the instrument.'],
                   'colnames' : ['w0','w1','flux','error','exptime','flags','instrument']}
                   
prenormed = ['mod_lya', 'mod_euv', 'cos_g130m', 'cos_g160m', 'cos_g230l',
             'sts_g230l', 'sts_g430l']

lyacut = [1213.5, 1218.0]

specstrings = ['_x1d', 'mod_euv', 'mod_lya', 'xmm', 'sx1', 'mod_phx']
#listed in normalization order
instruments = ['hst_cos_g130m','hst_cos_g160m','hst_cos_g230l','hst_sts_g230l',
               'hst_sts_e230m','hst_sts_e230h','hst_sts_e140m','hst_sts_g430l',
               'mod_euv_-----','xmm_mos_-----','mod_phx_-----','mod_lya_kevin']
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
        
    def add_custom_extraction(self, config, **kwds):
        d = {'config' : config, 'kwds' : kwds}
        self.custom_extractions.append(d)
        
    def add_reject(self, config, i=0):
        self.reject_specs.append([config, i])
        
    def save(self):
        path = db.settingspath(self.star)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)
        
def load(star):
    path = db.settingspath(star)
    with open(path) as f:
        d = json.load(f)
    ss = StarSettings(star)
    ss.custom_extractions = d['custom_extractions']
    ss.notes = d['notes']
    ss.reject_specs = d['reject_specs']
    return ss