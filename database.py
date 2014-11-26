# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 15:51:54 2014

@author: Parke
"""
import os
from astropy.io import fits
from numpy import array, nonzero, unique
import io
from muscles import utils

instruments = ['hst_cos_g130m','hst_cos_g160m','hst_cos_g230l','hst_sts_e140m',
               'hst_sts_e230m','hst_sts_e230h','hst_sts_g140m','hst_sts_g230l',
               'hst_sts_g430l','xmm_mos_-----','mod_lya_kevin','mod_euv_-----']
foldersbyband = {'u':'uv', 'v':'visible', 'x':'x-ray'}

def allspecfiles(target, folder='.'):
    """Find all the spectra for the target within the subdirectories of path
    using the file naming convention."""
    specstrings = ['_x1d', 'mod_euv', 'mod_lya', 'xmm', 'sx1']
    isspec = lambda name: any([s in name for s in specstrings])
    hastarget = lambda name: target in name
    
    subfolders = [folder]
    contents = [os.path.join(folder,p) for p in os.listdir(folder)]
    subfolders.extend(filter(os.path.isdir, contents))
    files = []
    for sf in subfolders:
        allfiles = os.listdir(sf)
        targetfiles = filter(hastarget, allfiles)
        specfiles = filter(isspec, targetfiles)
        specfiles = [os.path.join(folder, sf, f) for f in specfiles]
        files.extend(specfiles)
    
    return files

def parse_instrument(filename):
    """Parse out the instrument portion of a spectrum filename."""
    name = os.path.basename(filename)
    pieces = name.split('_')
    return '_'.join(pieces[1:4])

def coaddpath(specpath):
    """Construct standardized name for coadd FITS file within same directory as 
    specfile."""
    specdir = os.path.dirname(specpath)
    specname = os.path.basename(specpath)
    parts = specname.split('_')
    coaddname = '_'.join(parts[:5]) + '_coadd.fits'
    return os.path.join(specdir, coaddname)
    
def coadd_swap(specfile, datafolder='.'):
    """Search for coadds that cover groups of the specfiles and replace those
    groups with the coadds.
    """
    groups = utils.group_by_instrument(spectbls)
    for group in groups:
        coaddfile = find_coadd(group, datafolder)
        if len(coaddfile):
            group = [coaddfile]
    return 

def find_coadd(spectbls, datafolder='.', returntbl=False):
    #FIXME: not sure if I actually need this as it uses time to open tables
    #I should probably just do this at the file level
    """
    Look for a FITS file that is the coaddition of the provided spectlbs.
    Returns the coadd spectbl if it exists and it contains data from all of the
    provided spectbls otherwise returns none. 
    """
    #parse sourcefiles
    sourcefiles = []
    for spec in spectbls: sourcefiles.extend(spec.meta['sourcefiles'])
    
    #check for multiple configurations
    instruments = array(map(parse_instrument, sourcefiles))
    if any(instruments[:-1] != instruments[:-1]):
        return NotImplemented("...can't deal with different data sources.")
    
    #get the filename for one of the source files
    specname = spectbls[0].meta['sourcefiles'][0]
    
    #construct standard 
    band = specname[0] #should be 'u', 'v', or 'x'
    subfolder = foldersbyband[band]
    coaddname = coaddpath(specname)
    coaddfile = os.path.join(datafolder, subfolder, coaddname)
    
    try:
        coadd = io.read(coaddfile)
        
        #check that the coadd contains the same data as the spectbls
        #return none if any is missing
        csourcefiles = coadd.meta['sourcefiles']
        for sf in sourcefiles:
            if sf not in csourcefiles:
                return None
        return coadd if returntbl else return coaddfile
        
    #if the file doesn't exist, return None
    except IOError:
        return None

def auto_rename(folder):
    """
    Rename all of the files in the folder according to the standard naming
    convention as best as possible.
    """
    
    #find all the FITS files
    names = filter(lambda s: s.endswith('.fits'), os.listdir(folder))
    
    tele = None
    unchanged = []
    while len(names) > 0:
        name = names.pop(0)
        try:
            filepath = os.path.join(folder, name)
            hdr = fits.getheader(filepath)
            
            telekeys = ['telescop']
            for telekey in telekeys:
                try:
                    tele = hdr[telekey]
                    break
                except:
                    tele = None
            
            if tele is None:
                unchanged.append(name)
                continue
            if tele == 'HST':
                #using the x1d to get the appropriate info, rename all the files
                #from the same observation
                try:
                    root = hdr['rootname']
                except KeyError:
                    root = name[-18:-9]
                def isspec(s):
                    return (root + '_x1d.fits') in s or (root + '_sx1.fits') in s
                
                specfile = name if isspec(name) else filter(isspec, names)[0]
                xpath = os.path.join(folder,specfile)
                xhdr = fits.getheader(xpath)
                inst = xhdr['instrume']
                if inst == 'STIS': inst = 'STS'
                grating = xhdr['opt_elem']
                target = xhdr['targname']
                cenwave = xhdr['cenwave']
                band = 'U' if cenwave < 4000.0 else 'V'
                
                obsnames = filter(lambda s: root in s, names) + [name]
                for oname in obsnames:
                    try: 
                        names.remove(oname)
                    except ValueError:
                        pass
                    opath = os.path.join(folder, oname)
                    original_name = fits.getval(opath, 'filename')
                    newname = '_'.join([band, tele, inst, grating, target, 
                                        original_name])
                    os.rename(opath, os.path.join(folder, newname.lower()))           
        except:
            unchanged.append(name)
            continue
        
    if len(unchanged) > 0:
        print 'The following files could not be renamed:'
        for name in unchanged: print '    ' + name