# -*- coding: utf-8 -*-
"""
A collection of fucntions for visually inspecting the data and data products.

Created on Wed Dec 10 15:22:01 2014

@author: Parke
"""
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import database as db
import io
from plot import specstep
from numpy import mean
from os import path

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

def vetcoadds(star):
    """Plot the components of a coadded spectrum to check that the coadd agrees."""
    pass

def vetpanspec(star):
    """Plot unnormalized components of the panspec with the panspec to see that
    all choices were good. Phoenix spectrum is excluded because it is too big."""
    panspec = io.read(db.panpath(star))[0]
    specstep(panspec, color='k', err=True)
    files = db.panfiles(star)
    for f in files:
        if 'phx' in f: continue
        specs = io.read(f)
        for spec in specs:
            p = specstep(spec, alpha=0.3)[0][0]
            specstep(spec, color=p.get_color(), key='error', linestyle='--', 
                     alpha=0.3)
            x = (spec['w0'][0] + spec['w0'][-1])/2.0
            y = mean(spec['flux'])
            inst = db.parse_instrument(f)
            plt.text(x, y, inst, bbox={'facecolor':'w'}, ha='center', 
                     va='center')

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