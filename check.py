# -*- coding: utf-8 -*-
"""
A collection of fucntions for visually inspecting the data and data products.

Created on Wed Dec 10 15:22:01 2014

@author: Parke
"""
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
import database as db
import io
from plot import specstep
import numpy as np
from os import path
from spectralPhoton import image
from mypy.plotutils import pcolor_reg
from math import ceil, floor
from mypy.my_numpy import lace

stsfac = 2

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

def HSTcountregions(specfile):
    """
    Show where the spectrum was extracted in a 2d histogram of counts created
    from the tag or corrtag file of the same name.
    """
    
    if '_sts_' in specfile:
        #read data
        tagfile = specfile.replace('x1d', 'tag')
        td = fits.getdata(tagfile, 1)
        
        #make image
        __cnts2img(td['axis1'], td['axis2'])
        
        #get extraction region dimensions
        args = __stsribbons(specfile)
        N = args.pop()
        x = np.arange(1,N+1)*stsfac
        args.append(x)
        __plotribbons(*args)
        
    if '_cos_' in specfile:
        for seg in ['a', 'b']:
            #read data
            tagfile = specfile.replace('x1d', 'corrtag_'+seg)
            td = fits.getdata(tagfile, 1)
        
            #create image
            plt.figure()
            __cnts2img(td['xcorr'], td['ycorr'])
            
            #get extraction region dimensions
            args = __cosribbons(specfile, seg)
            N = args.pop()
            x = [1, N+1]
            args.append(x)
            __plotribbons(*args)

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
            y = np.mean(spec['flux'])
            inst = db.parse_instrument(f)
            plt.text(x, y, inst, bbox={'facecolor':'w'}, ha='center', 
                     va='center')

def HSTimgregions(specfile):
    """
    Show where the spectrum was extracted from the x2d of the same name.
    """
    #find 2d image file
    pieces = specfile.split('_')
    imgfiles = ['_'.join(pieces[:-1] + [s]) for s in ['crj.fits', 'sfl.fits']]
    imgfile = filter(path.exists, imgfiles)
    assert len(imgfile) == 1
    imgfile = imgfile[0]
    
    #get world coordinate system info
    with fits.open(imgfile) as imghdu:
        img = imghdu[1].data
        ihdr = imghdu[1].header
    w = WCS(ihdr)
    m, n = img.shape
    wave, _ = w.all_pix2world(np.arange(m), np.ones(n)*n//2, 0)
    ypix = np.arange(n)
    wave = wave*1e10 #to angstroms
    
    #plot 2d image file
    a = 0.3
    pcolor_reg(wave, ypix, img**a, cmap='Greys')
    plt.colorbar(label='flux**{:.2f}'.format(a))
    plt.ylabel('axis 1 (image)')
    plt.xlabel('axis 2 (wavelength, $\AA$)')
    
    if 'custom_spec' in specfile:
        spec = Table.read(specfile)
        smid, shgt, bhgt, bkoff = [spec.meta[s.upper()] for s in 
            ['traceloc','extrsize','bksize', 'bkoff']]
        b1mid, b2mid = smid - bkoff, smid + bkoff
        b1hgt, b2hgt = bhgt
        ribdims = smid, shgt, b1mid, b1hgt, b2mid, b2hgt
        x = wave[[0,-1]]
    elif '_sts_' in specfile:
        ribdims = __stsribbons(specfile)[:-1]
        spectbls = io.read(specfile)
        if len(spectbls) > 1:
            raise NotImplementedError('')
        else:
            spectbl = spectbls[0]
        x = np.append(spectbl['w0'], spectbl['w1'][-1])
    elif '_cos_' in specfile:
        ribdims = __cosribbons(specfile)[:-1]
        x = wave[[0,-1]]
        
    args = ribdims + [x]
    __plotribbons(*args)

def __cosribbons(specfile, seg):
    sh = fits.getheader(specfile, 1)
    smid = sh['sp_loc_'+seg]
    shgt = sh['sp_hgt_'+seg]
    b1mid, b2mid = sh['b_bkg1_'+seg], sh['b_bkg2_'+seg]
    b1hgt, b2hgt = sh['b_hgt1_'+seg], sh['b_hgt2_'+seg]
    N = sh['talen2']
    return [smid, shgt, b1mid, b1hgt, b2mid, b2hgt, N]
    
def __stsribbons(specfile):
    sd = fits.getdata(specfile, 1)
    smid = sd['extrlocy']*stsfac
    M, N = smid.shape
    b1mid, b2mid = [smid + sd[s][:, np.newaxis]*stsfac for s in 
                    ['bk1offst', 'bk2offst']]
    shgt, b1hgt, b2hgt = [np.outer(sd[s], np.ones(N))*stsfac for s in 
                    ['extrsize', 'bk1size','bk2size']]
    return [smid, shgt, b1mid, b1hgt, b2mid, b2hgt, N]
        
def __cnts2img(x,y):
    minx, maxx = floor(np.min(x)), ceil(np.max(x))
    miny, maxy = floor(np.min(y)), ceil(np.max(y))
    xbins, ybins = np.arange(minx, maxx+1), np.arange(miny, maxy+1)
    image(x, y, bins=[xbins, ybins], scalefunc='auto', cmap='Greys')

def __plotribbons(smid, shgt, b1mid, b1hgt, b2mid, b2hgt, x):
        triplets = [[smid, shgt, 'g'], [b1mid, b1hgt, 'r'], [b2mid, b2hgt, 'r']]
        for m, h, c in triplets:
            __plotribbon(m, h, c, x)

def __plotribbon(mid, hgt, color, x):
    #get limits
    lo, hi = mid - hgt//2, mid + hgt//2 + 1
    
    #lace if mid and hgt aren't scalar
    if not np.isscalar(mid):
        x = lace(x, x[1:-1])
        lo = lace(lo, lo, 1)
        hi = lace(hi, hi, 1)
    else:
        lo, hi = [lo], [hi]
        
    for llo, hhi in zip(lo, hi):
        plt.fill_between(x, hhi, llo, color=color, alpha=0.5)