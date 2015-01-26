# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 15:51:54 2014

@author: Parke
"""
import os
from astropy.io import fits
from pandas import read_pickle
from mypy.my_numpy import midpts
from numpy import array, nonzero, unique, argsort, arange, hstack, logical_and, nan, isnan 
import io, settings
from itertools import product as iterproduct
from urllib import urlretrieve
from math import ceil

datapath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Data'
productspath = r'C:\Users\Parke\Documents\Grad School\MUSCLES\Products'
codepath = r'C:\Users\Parke\Google Drive\Python\muscles'
root = r'C:\Users\Parke\Google Drive\Grad School\PhD Work\MUSCLES'

# -----------------------------------------------------------------------------
# PHOENIX DATABASE
phxpath = os.path.join(datapath, 'phoenix')
phxTgrid = hstack([arange(2300,7000,100),
                   arange(7000,12001,200)])
phxggrid = arange(0.0, 6.1, 0.5)
phxZgrid = hstack([arange(-4.0, -2.0, 1.0),
                   arange(-2.0, 1.1, 0.5)])
phxagrid = arange(-0.2, 1.3, 0.2)
phxgrids = [phxTgrid, phxggrid, phxZgrid, phxagrid]
phxwave = fits.getdata(os.path.join(phxpath, 'wavegrid_hires.fits'))
phxwave = hstack([[499.95], midpts(phxwave), [54999.875]])

def fetchphxfiles(Trng=[2500,3500], grng=[4.0,5.5], FeHrng=[0.0, 0.0], 
                  aMrng=[0.0, 0.0], repo=phxpath):
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
        
def phxspecpath(star):
    name = 'r_mod_phx_-----_{}_interpolated.fits'.format(star)
    return os.path.join(datapath, 'ir', name)

# -----------------------------------------------------------------------------
# STELLAR PROPERTIES DATABASE

proppath = os.path.join(root, 'star_properties.p')
refpath = os.path.join(root, 'star_properties_references.p')
errpospath = os.path.join(root, 'star_properties_errors_pos.p')
errnegpath = os.path.join(root, 'star_properties_errors_neg.p')

props = read_pickle(proppath)
proprefs = read_pickle(refpath)
properrspos =read_pickle(errpospath)
properrsneg = read_pickle(errnegpath)

stars = list(props.index)

def __setormask(tbl, path, star, prop, value):
    if value is None:
        isstr = isinstance(tbl[prop][star], basestring)
        tbl[prop][star] = '' if isstr else nan
    else:
        tbl[prop][star] = value
    tbl.to_pickle(path)
    
def setprop(star, prop, value, poserr=None, negerr=None, ref=None):
    setit = lambda tbl, path, value: __setormask(tbl, path, star, prop, value)
    setit(props, proppath, value)
    setit(properrspos, errpospath, poserr)
    setit(properrsneg, errnegpath, negerr)
    setit(proprefs, refpath, ref)
    
def printprop(star, name):
    tbls = [props, properrsneg, properrspos, proprefs]
    x, en, ep, ref = [t[name][star] for t in tbls]
    if ref == '': ref = 'no reference'
    print '{:f} -{:f} +{:f}\t{}'.format(x, en, ep, ref)
    
def phxinput(star):
    Teff = props['Teff'][star]
    kwds = {}
    for key in ['logg', 'FeH', 'aM']:
        x = props[key][star]
        if ~isnan(x):
            kwds[key] = x
    return Teff, kwds

# -----------------------------------------------------------------------------
# SPECTRAL DATA ORGANIZATION

def findfiles(path, substrings):
    """Look for a files in directory at path that contains ALL of the strings
    in substrings in its filename."""
    def good(name):
        hasstring = [(s in name) for s in substrings]
        return all(hasstring)
    return filter(good, os.listdir(path))

def findsimilar(specfile, newstring):
    """Find a file with the same identifier as sepcfile, but that also contains
    newstring in the file name. For example, find the the coadd version of the
    u_hst_cos_g130m_gj832 observation."""
    base = parse_id(specfile)
    dirname = os.path.dirname(specfile)
    names = findfiles(dirname, [base, newstring])
    paths = [os.path.join(dirname, n) for n in names]
    return paths
    
def configfiles(star, configstring, folder=datapath):
    """Find the spectra for the star that match configstring."""
    allfiles = allspecfiles(star, folder=folder)
    return filter(lambda f: configstring in f, allfiles)

def allspecfiles(star, folder=datapath):
    """Find all the spectra for the star within the subdirectories of path
    using the file naming convention."""
    isspec = lambda name: any([s in name for s in settings.specstrings])
    hasstar = lambda name: star in name
    
    subfolders = [folder]
    contents = [os.path.join(folder,p) for p in os.listdir(folder)]
    subfolders.extend(filter(os.path.isdir, contents))
    subfolders = filter(lambda f: 'phoenix' not in f, subfolders)
    files = []
    for sf in subfolders:
        allfiles = os.listdir(sf)
        starfiles = filter(hasstar, allfiles)
        specfiles = filter(isspec, starfiles)
        specfiles = [os.path.join(folder, sf, f) for f in specfiles]
        files.extend(specfiles)
    
    return files
    
def specfilegroups(star, folder=datapath):
    """Return a list of groups of files from the same instrument for
    instruments that have more than one file."""
    allfiles = allspecfiles(star, folder=folder)
    filterfiles = lambda s: filter(lambda ss: s in ss, allfiles)
    files = map(filterfiles, settings.instruments)
    files = filter(lambda x: len(x) > 1, files)
    return files
    
def panfiles(star, folder=datapath):
    """Return the files for the spectra to be spliced into a panspectrum,
    replacing "raw" files with coadds and custom extractions as appropriate
    and ordering according to how the spectra should be normalized."""
    
    allfiles = allspecfiles(star, folder=folder)
    use = lambda name: any([s in name for s in settings.instruments])
    allfiles = filter(use, allfiles)
    filterfiles = lambda s: filter(lambda ss: s in ss, allfiles)
    files = map(filterfiles, settings.instruments)
    files = reduce(lambda x,y: x+y, files)
    
    #sub in custom extractions
    files = sub_coaddfiles(files)
    files = sub_customfiles(files)
    
    return files

def parse_info(filename, start, stop):
    """Parse out the standard information bits from a muscles filename."""
    name = os.path.basename(filename)
    pieces = name.split('_')
    return '_'.join(pieces[start:stop])

def parse_instrument(filename):
    return parse_info(filename, 1, 4)
def parse_spectrograph(filename):
    return parse_info(filename, 2, 3)
def parse_band(filename):
    return parse_info(filename, 0, 1)
def parse_star(filename):
    return parse_info(filename, 4, 5)
def parse_id(filename):
    return parse_info(filename, 0, 5)
def parse_observatory(filename):
    return parse_info(filename, 1, 2)

def panpath(star):
    name = '-_msl_pan_-----_{}_panspec_native_resolution.fits'.format(star)
    return os.path.join(productspath, name)
    
def Rpanpath(star, R):
    name = ('-_msl_pan_-----_{}_panspec_constant_R={:d}.fits'
            ''.format(star, int(round(R))))
    return os.path.join(productspath, name)
    
def settingspath(star):
    return os.path.join(root, 'settings', star+'.json')
    
def getinsti(filename):
    try:
        i = settings.instruments.index(parse_instrument(filename))
    except ValueError:
        i = -99
    return i

def group_by_instrument(lst):
    """Group the spectbls by instrument, returning a list of the groups. Useful
    for coaddition. Preserves order. lst can be a list of filenames or a list
    of spectbls."""
    
    #get the unique instruments
    if type(lst[0]) is str:
        specfiles = lst
    else:
        specfiles = [spec.meta['FILENAME'] for spec in lst]
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
        csourcefiles = coadd.meta['SOURCEFILES']
        for sf in specfiles:
            if sf not in csourcefiles:
                return None
        return coaddfile
        
    #if the file doesn't exist, return None
    else:
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
                star = xhdr['targname']
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
                    newname = '_'.join([band, tele, inst, grating, star, 
                                        original_name])
                    os.rename(opath, os.path.join(folder, newname.lower()))           
        except:
            unchanged.append(name)
            continue
        
    if len(unchanged) > 0:
        print 'The following files could not be renamed:'
        for name in unchanged: print '    ' + name