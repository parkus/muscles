import io, db, rc, utils
from astropy.table import Table
import mypy.specutils as su

def phx_compare_single(star):
    pan = io.read(db.panpath(star))[0]
    xf = db.findfiles('ir', 'phx', star, fullpaths=True)
    phx = io.read(xf)[0]
    phx['flux'] *= pan['normfac'][-1]

    bands = [rc.fuv, rc.nuv, [rc.vis[0], 5700.]]
    (pff, pfe) , (pnf, pne), (pvf, pve) = [utils.flux_integral(pan, *b) for b in bands]
    (xff, _) , (xnf, _), (xvf, _) = [utils.flux_integral(phx, *b) for b in bands]

    return ((pff - xff)/pff, pfe/pff), ((pnf - xnf)/pnf, pne/pnf), ((pvf - xvf)/pvf, pve/pvf)

def phx_compare_table():
    tbl = Table(names=['star', 'fuv', 'fuv err', 'nuv', 'nuv err', 'vis', 'vis err'], dtype=['S10'] + ['f4']*6)

    for star in rc.observed:
        (fuv, fuv_err), (nuv, nuv_err), (vis, vis_err) = phx_compare_single(star)
        tbl.add_row([star, fuv, fuv_err, nuv, nuv_err, vis, vis_err])

    return tbl


def fuv_cont_spec(star):
    """Get just the continuum flux regions of a star's panspec."""
    spec = io.readpan(star)
    return utils.keepranges(spec, rc.contbands, ends='exact')


def fuv_cont_flux(star, bolo=True, order=2):
    """Compute the FUV continuum flux by fitting a quadratic. Compare to the total FUV line flux, Lya flux,
    and line - Lya flux. Return values in that order.
    """
    contspec = fuv_cont_spec()
    if bolo:
        bolo = utils.bolo_integral(star)
        contspec['flux'] /= bolo
        contspec['error'] /= bolo

    wbins, f, e = utils.wbins(spec), spec['flux'], spec['error']
    fit = su.polyfit()
