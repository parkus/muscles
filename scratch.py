import rc, io, db, utils, reduce as red
from plot import specstep as ss
from astropy import table
from matplotlib import pyplot as plt
import os

mlstars = rc.observed[:]
mlstars.remove('gj551')

# ----------------------------------------------------------------------------------------------------------------------
# COMPARE V20 AND V10 SPECTRA

def compare_v20_v10(stars=None, plot=False):

    if stars is None:
        stars = mlstars

    oldfile_template = '/Users/rolo7566/Datasets/MUSCLES/products/archive/panspecs v10/p_msl_pan_-----_{}_panspec_native_resolution.fits'
    hlspfile_template = '/Users/rolo7566/Google Drive/Grad School/PhD Work/muscles/share/hlsp/archive/hlsp_muscles_multi_multi_{}_broadband_v10_var-res-sed.fits'
    compare_template = '/Users/rolo7566/Google Drive/Grad School/PhD Work/muscles/scratchwork/v10 - v20 comparisons/{' \
                       '} v20 vs v10 ratio.fits'

    for star in stars:
        p = io.readpan(star)
        op, = io.read(oldfile_template.format(star))
        hp, = io.read(hlspfile_template.format(star))
        op = red.splice(op, utils.keepranges(hp, 100., 1169.))
        p, op = [utils.keepranges(spec, 0, 10000.) for spec in [p, op]]
        cs = utils.compare_specs(p, op, savetxt=compare_template.format(star))

        cf = compare_template.format(star)
        cs.write(cf, overwrite=True)

        if plot:
            plt.figure()
            ss(cs, key='ratio')
            plt.title(star)
            plt.xlabel('Wavelength $\AA$')
            plt.ylabel('Ratio of v20 to v10 SED')


# ----------------------------------------------------------------------------------------------------------------------
# COMPARE V20 AND V10 PHX NORMS AND GET PHOTOMETRY REFERENCES

def phx_norm_compare():
    oldfile_template = '/Users/rolo7566/Datasets/MUSCLES/products/archive/panspecs v10/p_msl_pan_-----_{}_panspec_native_resolution.fits'

    for star in mlstars:
        op, = io.read(oldfile_template.format(star))
        ofac = op['normfac'][-1]
        p = io.readpan(star)
        fac = p['normfac'][-1]
        chng = abs(fac/ofac - 1)*100.
        name = rc.starprops['name txt'][star]
        print '{:8s} | {:6.1f}'.format(name, chng)

# ----------------------------------------------------------------------------------------------------------------------
# plot Lya splice for all stars

def lya_splices(stars='all'):
    if stars=='all':
        stars = rc.stars[:11]
        stars.remove('gj551')
    dw = 0.05
    for star in stars:
        pan = io.readpan(star)
        pan = utils.keepranges(pan, 1100, 1300)
        pan = utils.evenbin(pan, dw)

        sf = db.findfiles('u', star, 'coadd', 'cos_g130m')
        spec, = io.read(sf)
        spec = utils.evenbin(spec, dw)

        lf = db.findfiles('u', star, 'mod', 'lya')
        lya, = io.read(lf)

        plt.figure()
        [ss(s, err=False) for s in [pan, spec, lya]]
        plt.xlim(1210, 1222)
        up, _ = utils.flux_integral(spec, 1217, 1220)
        plt.ylim(0, up*4)
        plt.legend(['pan', 'cos', 'lya'], loc='best')
        plt.savefig(os.path.join(rc.scratchpath, 'lya splices', '{} linear.pdf'.format(star)))

        mx = spec['flux'].max()
        plt.ylim(mx/1e7, mx)
        plt.yscale('log')
        plt.savefig(os.path.join(rc.scratchpath, 'lya splices', '{} log.pdf'.format(star)))
