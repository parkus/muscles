# -*- coding: utf-8 -*-
"""
A collection of fucntions for visually inspecting the data and data products.

Created on Wed Dec 10 15:22:01 2014

@author: Parke
"""
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits

def cyclespec(files):
    plt.ioff()
    for f in files:
        specs = io.read(f)
        for spec in specs:
            specstep(spec)
        plt.title(path.basename(f))
        plt.xlabel('Wavelength [$\AA$]')
        plt.ylabel('Flux [erg/s/cm$^2$/$\AA$]')
        plt.show()
    plt.ion()

def countregions():
    """
    Show where the spectrum was extracted in a 2d histogram of counts.
    """
    pass

def x2dregions(specfile):
    """
    Show where the spectrum was extracted from the x2d of the same name.
    """
    x2dfile = specfile.replace('custom_spec', 'x2d')
    spec = Table.read(specfile)
    f = fits.getdata(x2dfile, 1)
    
    a = 0.3
    plt.imshow(f**a)
    
    plt.colorbar(label='flux**{:.2f}'.format(a))
    
    plt.xlabel('axis 1 (image)')
    plt.ylabel('axis 2 (wavelength)')
    
    ys, ds, db, boff = [spec.meta[s.upper()] for s in 
                        ['traceloc','extrsize','bksize', 'bkoff']]
    
    for y in [ys-ds/2.0, ys+ds/2.0]:
        sline = plt.axhline(y,color='k')
        
    for y in [ys-boff-db/2.0, ys-boff+db/2.0, ys+boff-db/2.0, ys+boff+db/2.0]:
        bline = plt.axhline(y,color='k',linestyle='--')
        
    plt.legend((sline, bline), ('signal','background'))