# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 15:51:54 2014

@author: Parke
"""
import os
from astropy.io import fits

def auto_rename(folder):
    """
    Rename all of the files in the folder according to the standard naming
    convention as best as possible.
    """
    
    #find all the FITS files
    names = filter(lambda s: s.endswith('.fits'), os.listdir(folder))
    
    tele = None
    unchanged = []
    for name in names:
#        try:
        filepath = os.path.join(folder, name)
        hdr = fits.getheader(filepath)
        
        telekeys = ['telescop']
        for telekey in telekeys:
            try:
                tele = hdr[telekey]
            except:
                continue
        
        if tele is None:
            unchanged.append(name)
            continue
        if tele == 'HST':
            #using the x1d to get the appropriate info, rename all the files
            #from the same observation
            root = name[:9]
            x1dfile = filter(lambda s: (root + '_x1d.fits') in s, names)[0]
            xpath = os.path.join(folder,x1dfile)
            xhdr = fits.getheader(xpath)
            inst = xhdr['instrume']
            if inst == 'STIS': inst = 'STS'
            grating = xhdr['opt_elem']
            target = xhdr['targname']
            cenwave = xhdr['cenwave']
            band = 'U' if cenwave < 4000.0 else 'V'
            
            obsnames = filter(lambda s: s.count(root), names)
            for oname in obsnames:
                names.remove(oname)
                opath = os.path.join(folder, oname)
                original_name = fits.getval(opath, 'filename')
                newname = '_'.join([band, tele, inst, grating, target, 
                                    original_name])
                os.rename(opath, os.path.join(folder, newname))
                    
#        except:
#            unchanged.append(name)
#            continue
    if len(unchanged) > 0:
        print 'The following files could not be renamed:'
        for name in unchanged: print '    ' + name