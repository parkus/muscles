# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 15:51:54 2014

@author: Parke
"""
import os
from astropy.io import fits
from my_numpy import midpts
from numpy import array, nonzero, unique, argsort, arange, hstack, logical_and
import io
from itertools import product as iterproduct
from urllib import urlretrieve
from math import ceil

datapath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Data'

instruments = ['hst_cos_g130m','hst_cos_g160m','hst_cos_g230l','hst_sts_e140m',
               'hst_sts_e230m','hst_sts_e230h','hst_sts_g140m','hst_sts_g230l',
               'hst_sts_g430l','xmm_mos_-----','mod_lya_kevin','mod_euv_-----',
               'mod_phx_-----']
foldersbyband = {'u':'uv', 'v':'visible', 'x':'x-ray'}

phxTgrid = hstack([arange(2300,7000,100),
                   arange(7000,12001,200)])
phxggrid = arange(0.0, 6.1, 0.5)
phxZgrid = hstack([arange(-4.0, -2.0, 1.0),
                   arange(-2.0, 1.1, 0.5)])
phxagrid = arange(-0.2, 1.3, 0.2)
phxgrids = [phxTgrid, phxggrid, phxZgrid, phxagrid]
phxwave = fits.getdata(os.path.join(datapath, 'phoenix/wavegrid_hires.fits'))
phxwave = hstack([[499.95], midpts(phxwave), [54999.875]])

def findfiles(path, substrings):
    """Look for a file in directory at path that contains ALL of the strings
    in substrings in its filename."""
    def good(name):
        hasstring = [(s in name) for s in substrings]
        return all(hasstring)
    return filter(good, os.listdir(path))

def findsimilar(specfile, newstring):
    """Find a file with the same identifier as sepcfile, but that also contains
    newstring int he file name. For example, find the the coadd version of the
    u_hst_cos_g130m_gj832 observation."""
    base = parse_id(specfile)
    dirname = os.path.dirname(specfile)
    names = findfiles(dirname, [base, newstring])
    paths = [os.path.join(dirname, n) for n in names]
    return paths

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

def parse_info(filename, start, stop):
    """Parse out the standard information bits from a muscles filename."""
    name = os.path.basename(filename)
    pieces = name.split('_')
    return '_'.join(pieces[start:stop])

def parse_instrument(filename):
    return parse_info(filename, 1, 4)
def parse_band(filename):
    return parse_info(filename, 0, 1)
def parse_star(filename):
    return parse_info(filename, 4, 5)
def parse_id(filename):
    return parse_info(filename, 0, 5)

def group_by_instrument(lst):
    """Group the spectbls by instrument, returning a list of the groups. Useful
    for coaddition. Preserves order. lst can be a list of filenames or a list
    of spectbls."""
    
    #get the unique instruments
    if type(lst[0]) is str:
        specfiles = lst
    else:
        specfiles = [spec.meta['filename'] for spec in lst]
    allinsts = array(map(parse_instrument, specfiles))
    insts, ind = unique(allinsts, return_index=True)
    insts = insts[argsort(ind)]
    
    #group em
    groups = []
    for inst in insts:
        use = nonzero(allinsts == inst)[0]
        specgroup = [lst[i] for i in use]
        groups.append(specgroup)
    
    return groups

def coaddpath(specpath):
    """Construct standardized name for coadd FITS file within same directory as 
    specfile."""
    specdir = os.path.dirname(specpath)
    specname = os.path.basename(specpath)
    parts = specname.split('_')
    coaddname = '_'.join(parts[:5]) + '_coadd.fits'
    return os.path.join(specdir, coaddname)
    
def sub_coaddfiles(specfiles):
    """Replace any group of specfiles from the same instrument with a coadd
    file that includes data from all spectra in that group if one exists in the
    same directory.
    """
    groups = group_by_instrument(specfiles)
    result = []
    for group in groups:
        coaddfile = find_coaddfile(group)
        if coaddfile is not None:
            result.append(coaddfile)
        else:
            result.extend(group)
    return result
    
def sub_customfiles(specfiles):
    """Replace any file with a custom extraction file for the same instrument
    if one exists in the same directory."""
    result = []
    for name in specfiles:
        customfiles = findsimilar(name, 'custom')
        if len(customfiles) > 1:
            raise ValueError('Multiple matching files.')
        elif len(customfiles) == 1:
            result.append(customfiles[0])
        else:
            result.append(name)
    return result

def find_coaddfile(specfiles):
    """
    Look for a file that is the coaddition of the provided spectlbs.
    Returns the filename if it exists and it contains data from all of the
    provided spectbls, otherwise returns none. 
    """
    #check for multiple configurations
    insts = array(map(parse_instrument, specfiles))
    if any(insts[:-1] != insts[:-1]):
        return NotImplemented("...can't deal with different data sources.")
    
    coaddfile = coaddpath(specfiles[0])
    if os.path.isfile(coaddfile):
        coadd, = io.read(coaddfile)
        
        #check that the coadd contains the same data as the spectbls
        #return none if any is missing
        csourcefiles = coadd.meta['sourcefiles']
        for sf in specfiles:
            if sf not in csourcefiles:
                return None
        return coaddfile
        
    #if the file doesn't exist, return None
    else:
        return None

def fetchphxfiles(Trng=[2500,3500], grng=[4.0,5.5], FeHrng=[0.0, 0.0], 
                  aMrng=[0.0, 0.0], repo=datapath+'\phoenix'):
    """
    Download all Phoenix spectra covering the provided ranges. Does not
    re-download files that already exist in the directory.
    
    Default values are from UV variability sample properties.
    """
    def inclusive(grid, rng):
        use = logical_and(grid >= rng[0], grid <= rng[1])
        return grid[use]
    grids = map(inclusive, phxgrids, [Trng, grng, FeHrng, aMrng])
    
    combos = iterproduct(*grids)
    paths = []
    for combo in combos:
        locpath = io.phxpath(*combo, repo=repo)
        if not os.path.exists(locpath):
            paths.append((locpath, io.phxpath(*combo, repo='ftp')))
    
    N = len(paths)
    datasize = N*6.0/1024.0
    print ('Beginning download of {} files, {:.3f} Gb. Ctrl-C to stop.'
           ''.format(N,datasize))
    print 'Progress bar (- = 5%):\n'
    
    Nbar = ceil(N/20.0)
    for i,(loc, ftp) in enumerate(paths):
        if i % Nbar == 0:
            print '-',
        urlretrieve(ftp,loc)

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