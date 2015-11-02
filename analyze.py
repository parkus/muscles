from astropy.table import Table

import io
import db
import rc
import utils
import mypy.specutils as su
import numpy as np


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


def dissoc_spec(star, species):
    pan = io.readpan(star)
    pan = utils.add_photonflux(pan)
    bolo = utils.bolo_integral(star)  # erg s-1 cm-2

    xtbl = io.read_xsections(species)

    w = (pan['w0'] + pan['w1']) / 2.0
    xi = np.interp(w, xtbl['w'], xtbl['x'], 0.0, 0.0)
    basespec = xi * pan['flux_photon'] / bolo # cm2 * AA-1 erg-1

    xspec_bolo = np.zeros_like(basespec)
    for i in range(xtbl.meta['Nbranches']):
        y = xtbl['y_{}'.format(i)]
        yi = np.interp(w, xtbl['w'], y, 0.0, 0.0)
        xspec_bolo += basespec*y

    dw = pan['w1'] - pan['w0']
    dissoc_bolo = np.sum(dw * xspec_bolo) # cm2 erg-1
    xspec_sol = xspec_bolo * rc.insolation  # s-1 AA-1
    dissoc_sol = dissoc_bolo * rc.insolation # s-1

    return xspec_bolo, xspec_sol, dissoc_bolo, dissoc_sol



