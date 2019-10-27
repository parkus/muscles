from __future__ import division, print_function, absolute_import

from astropy.table import Table

from . import io
from . import db
from . import rc
from . import utils
import mypy.specutils as su
import mypy.my_numpy as mnp
import numpy as np
import astropy.units as u
import astropy.constants as c


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


def fuv_cont_stats(star):
    """
    Get stats on FUV continuum flux:
        - avg flux
        - avg flux error
        - raito of FUV continuum to total flux in the FUV assuming flat continuum
        - error on ratio
    """
    pan = io.readpan(star)
    cont = utils.keepranges(pan, rc.contbands, ends='exact')
    dw = cont['w1'] - cont['w0']

    # assume flat continuum
    # Fcont_avg = np.sum(cont['flux'] * dw)/np.sum(dw)
    # Fcont_avg_err = mnp.quadsum(cont['error'] * dw)/np.sum(dw)
    # dw_fuv = rc.fuv[1] - rc.fuv[0]
    # Fall_FUV, Fall_FUV_err = utils.flux_integral(pan, *rc.fuv)
    # ratio = Fcont_avg * dw_fuv / Fall_FUV
    # ratio_err = abs(ratio)*np.sqrt((Fall_FUV_err/Fall_FUV)**2 + (Fcont_avg_err/Fcont_avg)**2)

    # just do continuum actual measured, ignore "in-between" continuumFcont_avg
    Fcont  = np.sum(cont['flux'] * dw)
    Fcont_err = mnp.quadsum(cont['error'] * dw)
    Fall_FUV, Fall_FUV_err = utils.flux_integral(pan, cont['w0'][0], cont['w1'][-1])
    ratio = Fcont / Fall_FUV
    ratio_err = abs(ratio)*np.sqrt((Fall_FUV_err/Fall_FUV)**2 + (Fcont_err/Fcont)**2)
    return Fcont, Fcont_err, ratio, ratio_err


def dissoc_spec(star_or_spec, species_list, dissoc_only=True):
    if not hasattr(species_list, '__iter__'): species_list = [species_list]
    if type(star_or_spec) is str:
        star = star_or_spec
        spec = io.readpan(star)
    else:
        spec = star_or_spec
        star = spec.meta['STAR']
    spec = utils.add_photonflux(spec)
    bolo = utils.bolo_integral(spec) if 'pan' in spec.meta['NAME'] else utils.bolo_integral(star) # erg s-1 cm-2

    result = []
    for species in species_list:
        xtbl = io.read_xsections(species, dissoc_only=dissoc_only)
        w0, w1 = utils.wbins(spec).T

        w = (w0 + w1) / 2.0
        xsum = sumdissoc(xtbl)
        xsumi = np.interp(w, xtbl['w'], xsum, np.nan, np.nan)
        xspec_bolo = spec['flux_photon'] * xsumi / bolo

        dw = w1 - w0
        dissoc_bolo = np.nansum(dw * xspec_bolo) # cm2 erg-1

        dtbl = Table(data=[w0, w1, xspec_bolo], names=['w0', 'w1', 'diss'])
        dtbl['diss'].units = u.cm**2 / u.AA / u.erg
        dtbl['w0'].units = dtbl['w1'].units = u.AA
        result.append([dtbl, dissoc_bolo])

    return result


def cum_dissoc_spec(star_or_spec, species, dissoc_only=True, normed=True):
    dspec = dissoc_spec(star_or_spec, species, dissoc_only=dissoc_only)[0][0]
    dw = dspec['w1'] - dspec['w0']
    temp = dspec['diss'].copy()
    temp[np.isnan(temp)] = 0.0
    cumspec = np.cumsum(temp * dw) * rc.insolation
    cumspec = np.insert(cumspec, 0, 0.0)
    if normed:
        cumspec /= cumspec[-1]
    return cumspec


def dissoc_ratio(spec, species, band1, band2, dissoc_only=True):
    """Returns ratio of dissociations due to flux in band1 over flux in band2."""
    cspec = cum_dissoc_spec(spec, species, dissoc_only=dissoc_only)
    we = utils.wedges(spec)
    integral = lambda band: np.diff(np.interp(band, we, cspec))
    I1, I2 = list(map(integral, [band1, band2]))
    return I1/I2


def sumdissoc(xtbl):
    sumx = np.zeros_like(xtbl['x'])
    for i in range(xtbl.meta['Nbranches']):
        y = xtbl['y_{}'.format(i)]
        sumx += xtbl['x'] * y
    return sumx


def fluxall(band=rc.fuv):
    Is = []
    for star in rc.observed:
        spec = io.readpan(star)
        I = utils.flux_integral(spec, *band)
        Is.append(I)
    return list(zip(*Is))
