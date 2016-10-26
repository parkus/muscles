import rc, io, db, utils, reduce as red
from plot import specstep as ss
from astropy import table
from matplotlib import pyplot as plt

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