from astropy.table import Table

import io
import db
import rc
import utils
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
    Fcont_avg = np.sum(cont['flux'] * dw)/np.sum(dw)
    Fcont_avg_err = mnp.quadsum(cont['error'] * dw)/np.sum(dw)
    dw_fuv = rc.fuv[1] - rc.fuv[0]
    Fall_FUV, Fall_FUV_err = utils.flux_integral(pan, *rc.fuv)
    ratio = Fcont_avg * dw_fuv / Fall_FUV
    ratio_err = abs(ratio)*np.sqrt((Fall_FUV_err/Fall_FUV)**2 + (Fcont_avg_err/Fcont_avg)**2)
    return Fcont_avg, Fcont_avg_err, ratio, ratio_err


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
        minipan = utils.keepranges(spec, xtbl['w'][0] - 10., xtbl['w'][-1] + 10.)
        w0, w1 = utils.wbins(minipan).T

        w = (w0 + w1) / 2.0
        xsum = sumdissoc(xtbl)
        xsumi = np.interp(w, xtbl['w'], xsum, np.nan, np.nan)
        xspec_bolo = minipan['flux_photon'] * xsumi / bolo

        dw = w1 - w0
        dissoc_bolo = np.nansum(dw * xspec_bolo) # cm2 erg-1

        dtbl = Table(data=[w0, w1, xspec_bolo], names=['w0', 'w1', 'diss'])
        dtbl['diss'].units = u.cm**2 / u.AA / u.erg
        dtbl['w0'].units = dtbl['w1'].units = u.AA
        result.append([dtbl, dissoc_bolo])

    return result


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
    return zip(*Is)
