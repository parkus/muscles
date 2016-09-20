import rc, io
import os
import numpy as np
from astropy.io import fits


def findfiles(path_or_band, *substrings, **kwargs):
    """Look for a files in directory at path that contains ALL of the strings
    in substrings in its filename. Add fullpaths=False if desired."""

    if not os.path.exists(path_or_band):
        band = path_or_band if len(path_or_band) > 1 else rc.bandmap[path_or_band]
        path_or_band = rc.datapath + '/' + band

    def good(name):
        hasstring = [(s in name) for s in substrings]
        return all(hasstring)

    files = filter(good, os.listdir(path_or_band))
    if 'fullpaths' in kwargs and kwargs['fullpaths'] == False:
        return files

    files = [os.path.join(path_or_band, f) for f in files]
    return files


def validpath(name):
    if os.path.exists(name):
        return name
    else:
        name = os.path.basename(name)
        band = name[0]
        folder = rc.bandmap[band]
        path = os.path.join(rc.datapath, folder, name)
        if path[-4:] != 'fits':
            path += '.fits'
        if not os.path.exists(path):
            raise IOError("Can't find file {} in the standard place ({})."
                          "".format(name, os.path.join(rc.datapath, folder)))
        else:
            return path

def findsimilar(specfile, newstring):
    """Find a file with the same identifier as sepcfile, but that also contains
    newstring in the file name. For example, find the the coadd version of the
    u_hst_cos_g130m_gj832 observation."""
    base = parse_id(specfile)
    dirname = os.path.dirname(specfile)
    names = findfiles(dirname, base, newstring)
    paths = [os.path.join(dirname, n) for n in names]
    return paths


def configfiles(star, configstring):
    """Find the spectra for the star that match configstring."""
    allfiles = allspecfiles(star)
    return filter(lambda f: configstring in f, allfiles)


def choosesourcespecs(specfiles):
    """Given a list of specfiles, remove coadds and replace originals
    with custom files."""
    # get rid of reduced files
    specfiles = filter(lambda s: not ('coadd' in s or 'custom' in s), specfiles)

    # remove any non-spec files
    specfiles = filter(isspec, specfiles)

    return specfiles


def sourcespecfiles(star, configstring):
    """Source spectrum files that conatin configstring."""
    return choosesourcespecs(configfiles(star, configstring))


def coaddfile(star, configstring):
    """The coadd file for a config and star."""
    allfiles = allspecfiles(star)
    f = filter(lambda f: configstring in f and 'coadd' in f, allfiles)
    if len(f) > 1:
        raise Exception('Multiple files found.')
    else:
        return f[0]


def customfile(star, configstring):
    """The custom extraction file for a config and star."""
    allfiles = allspecfiles(star)
    f = filter(lambda f: configstring in f and 'custom_spec' in f, allfiles)
    if len(f) > 1:
        raise Exception('Multiple files found.')
    else:
        return f[0]


isspec = lambda name: any([s in name for s in rc.specstrings])


def allspecfiles(star):
    """Find all the spectra for the star within the subdirectories of path
    using the file naming convention."""
    hasstar = lambda name: star in name

    folders = [os.path.join(rc.datapath, p) for p in rc.datafolders]
    files = []
    for sf in folders:
        allfiles = os.listdir(sf)
        starfiles = filter(hasstar, allfiles)
        specfiles = filter(isspec, starfiles)
        specfiles = [os.path.join(rc.datapath, sf, f) for f in specfiles]
        files.extend(specfiles)

    return files


def allsourcefiles(star):
    """All source spectrum files for a star."""
    allfiles = allspecfiles(star)
    return choosesourcespecs(allfiles)


def panfiles(star):
    """Return the files for the spectra to be spliced into a panspectrum,
    replacing "raw" files with coadds and custom extractions as appropriate
    and ordering according to how the spectra should be normalized."""

    allfiles = allsourcefiles(star)
    use = lambda name: any([s in name for s in rc.instruments])
    allfiles = filter(use, allfiles)
    filterfiles = lambda s: filter(lambda ss: s == parse_info(ss, 1, 4), allfiles)
    files = map(filterfiles, rc.instruments)
    files = reduce(lambda x, y: x + y, files)

    # sub in custom extractions
    files = sub_customfiles(files)
    files = sub_coaddfiles(files)

    # parse out lya file
    lyafile = filter(lambda f: 'mod_lya' in f, files)
    assert len(lyafile) <= 1
    if len(lyafile):
        lyafile = lyafile[0]
        files.remove(lyafile)
    else:
        lyafile = None

    return files, lyafile


def solarfiles(date):
    files = os.listdir(rc.solarpath)
    files = filter(lambda s: date in s, files)
    ufile = filter(lambda s: 'u' == s[0], files)[0]
    vfile = filter(lambda s: 'v' == s[0], files)[0]
    ufile, vfile = [os.path.join(rc.solarpath, f) for f in (ufile, vfile)]
    return ufile, vfile


def lyafile(star):
    """Find the file with the best Lya data for star."""
    files = findfiles('uv', star, 'sts', '140')
    files = filter(isspec, files)
    files = [os.path.join(rc.datapath, 'uv', f) for f in files]
    files = sub_customfiles(files)
    files = sub_coaddfiles(files)
    if len(files) > 1:
        raise ValueError('More than one file found:\n' + '\n\t'.join(files))
    else:
        return os.path.basename(files[0])


def parse_info(filename, start, stop):
    """Parse out the standard information bits from a muscles filename."""
    name = os.path.basename(filename)
    pieces = name.split('_')
    slc = slice(start, stop)
    return '_'.join(pieces[slc])


def parse_instrument(filename):
    return parse_info(filename, 1, 4)


def parse_spectrograph(filename):
    return parse_info(filename, 2, 3)


def parse_grating(filename):
    return parse_info(filename, 3, 4)


def parse_band(filename):
    return parse_info(filename, 0, 1)


def parse_star(filename):
    return parse_info(filename, 4, 5)


def parse_id(filename):
    return parse_info(filename, 0, 6)


def parse_observatory(filename):
    return parse_info(filename, 1, 2)


def parse_paninfo(filename):
    return parse_info(filename, 6, None)


def parse_name(filename):
    name = os.path.basename(filename)
    return '.'.join(name.split('.')[:-1])


def name2path(name):
    return os.path.join(rc.datapath, rc.bandmap[name[0]], name+'.fits')


def allpans(star):
    """All panspec files for a star."""
    allfiles = os.listdir(rc.productspath)
    identifier = '{}_panspec'.format(star)
    panfiles = filter(lambda s: identifier in s, allfiles)
    return [os.path.join(rc.productspath, pf) for pf in panfiles]


def panpath(star):
    """The native resolution panspec file for a star."""
    name = 'p_msl_pan_-----_{}_panspec_native_resolution.fits'.format(star)
    return os.path.join(rc.productspath, name)


def Rpanpath(star, R):
    """The constant R panspec file for a star."""
    name = ('p_msl_pan_-----_{}_panspec_constant_R={:d}.fits'
            ''.format(star, int(round(R))))
    return os.path.join(rc.productspath, name)


def dpanpath(star, dR):
    """The constant resolution (binsize) panspec file for a star."""
    name = ('p_msl_pan_-----_{}_panspec_constant_dR={:.1f} angstrom.fits'
            ''.format(star, float(dR)))
    return os.path.join(rc.productspath, name)


def getinsti(filename):
    """Returns the numeric identifier for the instrument that created a
    spectrum based on the filename."""
    return rc.getinsti(parse_instrument(filename))


def group_by_instrument(lst):
    """Group the spectbls by instrument, returning a list of the groups. Useful
    for coaddition. Preserves order. lst can be a list of filenames or a list
    of spectbls."""

    # get the unique instruments
    if type(lst[0]) is str:
        specfiles = lst
    else:
        specfiles = [spec.meta['FILENAME'] for spec in lst]
    allinsts = np.array(map(parse_instrument, specfiles))
    insts, ind = np.unique(allinsts, return_index=True)
    insts = insts[np.argsort(ind)]

    # group em
    groups = []
    for inst in insts:
        use = np.nonzero(allinsts == inst)[0]
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


def photometrypath(star):
    return os.path.join(rc.photometrypath, 'photometry_{}.vot'.format(star))


def photonpath(star, inst, seg=''):
    if seg == '':
        seg = '-'
    name = '_'.join([inst, seg, star, 'photons.fits'])
    return os.path.join(rc.photondir, name)


def flarepath(star, inst, label):
    inst = filter(lambda s: inst in s, rc.instruments)
    assert len(inst) == 1
    inst = inst[0]
    name = '_'.join([inst, star, label, 'flares'])
    return os.path.join(rc.flaredir, name + '.fits')


def hlsppath(name_or_star):

    if '_' not in name_or_star:
        star = name_or_star
        tel, inst, filt = 'multi', 'multi', 'broadband'
        product = 'var-res-sed'
    else:
        name = name_or_star
        star = parse_star(name)
        if 'panspec' in name:
            tel = 'multi'
            inst = 'multi'
            filt = 'broadband'
            if 'native' in name:
                product = 'var-res-sed'
            elif 'constant_dR' in name:
                product = 'const-res-sed'
            elif 'adaptive_oversampled' in name:
                product = 'adapt-const-res-sed'
            elif 'adaptive' in name:
                product = 'adapt-var-res-sed'
        else:
            tel, inst, filt = parse_instrument(name).split('_')
            tel = rc.HLSPtelescopes[tel].lower()
            inst = rc.HLSPinstruments[inst].lower()
            filt = rc.HLSPgratings[filt].lower()
            product = 'component-spec'

    if inst == 'polynomial-fit':
        return 'polynomial fits not separately saved'

    name = ('hlsp_muscles_{tel}_{inst}_{star}_{filter}_v{version}_{product}.fits'
            ''.format(tel=tel, inst=inst, star=star, filter=filt, version=rc.version, product=product))
    return os.path.join(rc.hlsppath, name)


def auto_rename(folder):
    """
    Rename all of the files in the folder according to the standard naming
    convention as best as possible.
    """

    # find all the FITS files
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
                # using the x1d to get the appropriate info, rename all the files
                # from the same observation
                try:
                    root = hdr['rootname']
                except KeyError:
                    root = name[-18:-9]

                def isspec(s):
                    return (root + '_x1d.fits') in s or (root + '_sx1.fits') in s

                #                specfile = name if isspec(name) else filter(isspec, names)[0]
                #                xpath = os.path.join(folder,specfile)
                #                xhdr = fits.getheader(xpath)
                inst = hdr['instrume']
                if inst == 'STIS': inst = 'STS'
                grating = hdr['opt_elem']
                star = hdr['targname']
                cenwave = hdr['cenwave']
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


def find_coaddfile(specfiles):
    """
    Look for a file that is the coaddition of the provided spectlbs.
    Returns the filename if it exists and it contains data from all of the
    provided spectbls, otherwise returns none.
    """
    # check for multiple configurations
    insts = np.array(map(parse_instrument, specfiles))
    if any(insts[:-1] != insts[:-1]):
        return NotImplemented("...can't deal with different data sources.")

    coaddfile = coaddpath(specfiles[0])
    if os.path.isfile(coaddfile):
        coadd, = io.read(coaddfile)

        # check that the coadd contains the same data as the spectbls
        # return none if any is missing
        csourcespecs = coadd.meta['SOURCESPECS']
        for sf in specfiles:
            if parse_name(sf) not in csourcespecs:
                return None
        return coaddfile

    # if the file doesn't exist, return None
    else:
        return None


def sub_coaddfiles(specfiles):
    """Replace any group of specfiles from the same instrument with a coadd
    file that includes data from all spectra in that group if one exists in the
    same directory.
    """
    groups = group_by_instrument(specfiles)
    result = []
    for group in groups:
        group = filter(lambda s: 'coadd' not in s, group)
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
            if customfiles[0] not in result:
                result.append(customfiles[0])
        else:
            result.append(name)
    return result


def coaddgroups(star, nosingles=False):
    """Return a list of groups of files that should be coadded (only HST files).
    Chooses the best source files and avoids dulicates."""
    allfiles = allsourcefiles(star)
    allfiles = sub_customfiles(allfiles)
    hstfiles = filter(lambda s: 'hst' in s, allfiles)
    filterfiles = lambda s: filter(lambda ss: s in ss, hstfiles)
    files = map(filterfiles, rc.instruments)
    files = filter(len, files)
    if nosingles:
        files = filter(lambda x: len(x) > 1, files)
    return files
