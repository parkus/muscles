# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:08:18 2015

@author: Parke
"""
import json
import database as db

dqmask = {'cos' : (64 | 2048 | 1 | 256 | 128 | 32 |16),
          'sts' : (1 | 2 | 4 | 128 | 256 | 512 | 4096 | 8192)}
          
normorder = ['cos', 'sts', 'xmm', 'mod_lya', 'mod_euv', 'mod_phx']

spectbl_format =  {'units' : ['Angstrom']*2 + ['erg/s/cm2/Angstrom']*2 + ['s','',''],
                   'dtypes' : ['f8']*5 + ['i2', 'i1'],
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
                   
prenormed = ['mod_lya', 'mod_euv']

lyacut = [1213.5, 1218.0]

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
        self.notes = []
        
    def add_custom_extraction(self, config, **kwds):
        d = {'config' : config, 'kwds' : kwds}
        self.custom_extractions.append(d)
        
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
    return ss