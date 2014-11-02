# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import numpy as np
from astropy.io import fits

def combine_specs(specfiles):
    
    configs = [s[3:16] for s in specfiles]
    configs = np.unique(configs)
    
    groups = [['HST COS G130M', 'HST COS G160M'], ['HST STS E140M', 'HST STIS E230M', ']
    
    for config in configs:
        