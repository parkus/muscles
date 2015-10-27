from os import path
from astropy import constants as const
import numpy as np
import io, db, rc, utils
from astropy.table import Table
import mypy.specutils as su
import pysynphot as psp
import sys
import cStringIO

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


def mag(star_or_spectbl, band='johnson,b', ref='ab'):
    """Computes magnitudes using STScI's synphot software. Band must follow the convention for that software,
    e.g. 'johnson,v', 'bessell,j'."""

    if type(star_or_spectbl) is str:
        spectbl = io.read(db.Rpanpath(star_or_spectbl, 10000))[0]
    else:
        spectbl = star_or_spectbl
    w = (spectbl['w0'] + spectbl['w1']) / 2.0
    f = spectbl['flux']

    if ref == 'ab':
        refspec = psp.FlatSpectrum(0.0, fluxunits='abmag')
    elif ref == 'vega':
        refspec = psp.Vega
    elif ref == 'st':
        refspec = psp.FlatSpectrum(0.0, fluxunits='stmag')
    else:
        raise ValueError('Unrecognized ref.')

    # suppress the annoying binset warnings that pop up and I think are not fatal
    temp = sys.stdout
    sys.stdout = cStringIO.StringIO()

    bp = psp.ObsBandpass(band)
    spec = psp.ArraySpectrum(w, f, fluxunits='flam', keepneg=True)
    obs = psp.Observation(spec, bp)
    refobs = psp.Observation(refspec, bp)
    mag = -2.5*np.log10(obs.integrate()/refobs.integrate())

    sys.stdout = temp

    return mag