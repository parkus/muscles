# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:42:13 2014

@author: Parke
"""

import os
from math import sqrt
from itertools import combinations_with_replacement as combos
from warnings import warn
import json

import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from scipy.stats import norm
import scipy.optimize as opt

import mypy.my_numpy as mnp
from mypy import specutils, pdfutils
import rc, utils, io, check, db
from spectralPhoton.hst import x2dspec
import spectralPhoton as sp
import matplotlib.pyplot as plt

colnames = rc.spectbl_format['colnames']
airglow_ranges = rc.airglow_ranges
safe_ranges = [0.0] + list(airglow_ranges.ravel()) + [np.inf]
safe_ranges = np.reshape(safe_ranges, [len(airglow_ranges) + 1, 2])


def theworks(star, newphx=False, silent=False):

    try:
        rc.loadsettings(star)
    except IOError:
        warn("No settings file found for {}. Initializing one.".format(star))
        sets = rc.StarSettings(star)
        sets.save()

    # interpolate and save phoenix spectrum
    if newphx:
        if not silent: print '\n\ninterpolating phoenix spectrum'
        auto_phxspec(star, silent=silent)
    else:
        if not silent: print '\n\nnot interpolating new phoenix spectrum bc you said not to'

    # make custom extractions
    if not silent: print '\n\nperforming any custom extractions'
    auto_customspec(star, silent=silent)

    # coadd spectra
    if not silent: print '\n\ncoadding spectra'
    auto_coadd(star, silent=silent)

    # make panspectrum
    if not silent: print '\n\nstitching spectra together'
    panspectrum(star, silent=silent)  # panspec and Rspec

    # write hlsp
    io.writehlsp(star, overwrite=True)


def adaptive_rebin_pans(star):
    pan = io.readpan(star)
    adapt = utils.killnegatives(pan, quickndirty=False)
    over = utils.evenbin(adapt, 1.0)
    name = pan.meta['NAME']
    adapt.meta['NAME'] = name.replace('native', 'adaptive')
    over.meta['NAME'] = name.replace('native', 'adaptive_oversampled')
    [io.writehlsp(spec, components=False) for spec in  [adapt, over]]


def panspectrum(star, savespecs=True, plotnorms=False, silent=False, phxnormerr='constSN'):
    """
    Coadd and splice the provided spectra into one panchromatic spectrum
    sampled at the native resolutions and constant R.

    Overlapping spectra will be normalized with the assumptions that they are
    listed in order of descending quality.
    """
    sets = rc.loadsettings(star)

    specs, lyaspec = io.read_panspec_sources(star)

    names = [s.meta['NAME'] for s in specs]

    # make sure all spectra are of the same star
    star = __same_star(specs)

    # make sure spectra are each from a single source
    for i, s in enumerate(specs):
        try:
            __same_instrument([s])
        except ValueError:
            raise ValueError('More than one instrument used in spectbl {}'.format(i))

    # carry out custom trims according to user-defined settings
    if not silent:
        print 'trimming spectra according to settings for {}'.format(star)
    for i in range(len(specs)):
        goodranges = sets.get_custom_range(names[i])
        if goodranges is not None:
            if not silent:
                print ('trimming spectra in {} to the ranges {}'
                       ''.format(names[i], goodranges))
            ends = 'loose' if 'cos_g130m' in names[i] else 'tight'
            specs[i] = utils.keepranges(specs[i], goodranges, ends=ends)

    # adjust wavelenghts as specified
    for i in range(len(specs)):
        offset = sets.get_wave_offset(names[i])
        if offset is not None:
            if not silent:
                print ('adjusting wavelengt of {} by {}'.format(names[i], offset))
            specs[i]['w0'] += offset
            specs[i]['w1'] += offset

    # remove airglow lines from COS
    if not silent:
        print '\n\tremoving airglow from G130M spectrum'
    for i in range(len(specs)):
        if 'cos_g130m' in names[i]:
            specs[i] = utils.keepranges(specs[i], safe_ranges)
            # CLOOGE: remove some of g140m or e140m so it isn't used from 1198-lya
        #        if 'sts_e140m' in names[i] or 'sts_g140m' in names[i]:
        #            keep = [[0.0, safe_ranges[3,0]], [safe_ranges[3,1], np.inf]]
        #            specs[i] = utils.keepranges(specs[i], keep, ends='loose')


    # for easily finding spectra in list
    def index(str):
        name = filter(lambda s: str in s, names)
        assert len(name) == 1
        return names.index(name[0])

    # normalize PHX to photometry
    if not silent: print 'normalizing phoenix to photometry'
    iphx = index('phx')
    phxnorm, phxerr = norm2photometry(specs[iphx], silent=silent, plotfit=False, clean=True, err=phxnormerr)
    specs[iphx]['flux'] *= phxnorm
    specs[iphx]['normfac'] = phxnorm
    rc.normfacs[star]['mod_phx_-----'] = phxnorm, phxerr

    # trim PHX models so they aren't used to fill small gaps in UV data
    if not silent:
        print '\n\ttrimming PHOENIX to 2500+'
    for i in range(len(specs)):
        if 'mod_phx' in names[i]:
            specs[i] = utils.split_exact(specs[i], 2500., 'red')

    # reorder spectra according to how they should be spliced
    order = sets.order if len(sets.order) else rc.default_order
    def order_index(spec):
        name = spec.meta['NAME']
        inst = db.parse_info(name, 1, 4)
        return order.index(inst)
    specs = sorted(specs, key=order_index)
    names = [spec.meta['NAME'] for spec in specs]

    # normalize and splice according to input order
    spec = specs[0]
    if not silent:
        print '\nstarting stitched spectrum with {}'.format(spec.meta['NAME'])
    for i in range(1, len(specs)):
        addspec = specs[i]
        name = names[i]

        if not silent:
            specrange = [addspec['w0'][0], addspec['w1'][-1]]
            print ''
            print 'splicing in {}, covering {:.1f}-{:.1f}'.format(name, *specrange)

        inst = db.parse_instrument(name)
        if not rc.dontnormalize(addspec):
            if inst in sets.weird_norm:
                refinst = sets.weird_norm[inst]
                if not silent:
                    print 'normalizing {} spec using the same factor as that used for the {} spec'.format(inst, refinst)
                refspec = filter(lambda spec: refinst in spec.meta['NAME'], specs)
                assert len(refspec) == 1
                refspec = refspec[0]
                normfac, normerr = refspec[0]['normfac'], np.nan
            else:
                overlap = utils.overlapping(spec, addspec)
                if not overlap:
                    normfac, normerr = 1.0, np.nan
                    if not silent:
                        print '\tno overlap, so won\'t normalize'
                if overlap:
                    if not silent:
                        print '\tnormalizing within the overlap'
                    normranges = sets.get_norm_range(name)
                    config = db.parse_info(name, 1, 4)
                    if config in rc.normranges:
                        if normranges is None:
                            normranges = rc.normranges[config]
                        else:
                            raise ValueError('Uh oh. Conflicting norm ranges. One in rc and one in star settings.')
                    if normranges is None:
                        normspec = addspec
                    else:
                        normspec = utils.keepranges(addspec, normranges)
                    safe = False if ('430' in name or '750' in name) else True
                    normfac, normerr = normalize(spec, normspec, silent=silent, safe=safe)
                    # HACK: phx plot breaks things, so I'm just not doing it for now
                if plotnorms and normfac != 1.0 and 'phx' not in name:
                    check.vetnormfacs(addspec, spec, normfac, normranges)

            addspec['flux'] *= normfac
            addspec['error'] *= normfac
            addspec['normfac'] = normfac
            specs[i] = addspec  # so i can use normalized specs later (lya)
            rc.normfacs[star][inst] = normfac, normerr
        else:
            if 'phx' not in name and 'g430' not in name:
                rc.normfacs[star][inst] = 1.0, np.nan
            if not silent: print '\twon\'t normalize, cuz you said not to'

        spec = smartsplice(spec, addspec, silent=silent)
    spec.meta['NAME'] = db.parse_name(db.panpath(star))
    if savespecs:
        with open(rc.normfac_file, 'w') as f:
            json.dump(rc.normfacs, f)

    # replace lya portion with model or normalized stis data
    if lyaspec is None:
        name = filter(lambda s: 'sts_g140m' in s or 'sts_e140m' in s, names)
        if len(name) > 1:
            raise Exception('More than one Lya stis file found.')
        ilya = names.index(name[0])
        lyaspec = specs[ilya]
        normfac = lyaspec['normfac'][0]
        if not silent:
            print ('replacing section {:.1f}-{:.1f} with STIS data from {lf}, '
                   'normalized by {normfac}'
                   ''.format(*rc.lyacut, lf=name, normfac=normfac))
    else:
        if not silent:
            print ('replacing section {:.1f}-{:.1f} with data from {lf}'
                   ''.format(*rc.lyacut, lf=lyaspec.meta['NAME']))
    lyaspec = utils.keepranges(lyaspec, rc.lyacut)
    spec = splice(spec, lyaspec)

    # fill any remaining gaps
    order, span = rc.gap_fit_order, rc.gap_fit_span
    if not silent:
        print ('filling in any gaps with an order {} polynomial fit to an '
               'area {}x the gap width'.format(order, span))
    spec = fill_gaps(spec, fill_with=order, fit_span=span, silent=silent,
                     mingapR=10.0)

    # resample at constant dw
    dw = rc.panres
    if not silent:
        print ('creating resampled panspec at dw = {:.1f} AA'.format(dw))
    dspec = utils.evenbin(spec, dw)

    # save to fits
    if savespecs:
        paths = [db.panpath(star), db.dpanpath(star, dw)]
        if not silent:
            print 'saving spectra to \n' + '\n\t'.join(paths)
        for s, path in zip([spec, dspec], paths):
            io.writefits(s, path, overwrite=True)

    return spec


def solarspec(date):
    """
    Read and splice together ultraviolet and visible solar spectra. These
    must be in the muscles/solar_data folder and appropriately named. This
    means if you want new ones you need to get them using LASP solar data
    portal thingy. Date format is yyyy-mm-dd.
    """
    ufile, vfile = db.solarfiles(date)
    u, v = [io.readcsv(f)[0] for f in (ufile, vfile)]
    w = 1850.0
    u = utils.split_exact(u, w, 'blue')
    v = utils.split_exact(v, w, 'red')
    return utils.vstack([u, v])


def normalize(spectbla, spectblb, worry=0.05, flagmask=False, silent=False, safe=True):
    """
    Normalize the spectrum b to spectrum a.

    Parameters
    ----------
    spectbla : muscles spectrum
        spectrum to normalize against
    spectblb : muscles spectrum
        spectrum to be normalized. wavelength and flux units must be the
        same in a and b
    worry : {float|False}
        consider errors in the overlapping area of the spectra. if the
        areas are consistent with being identical with probability p
        > worry, normalization will not occur. If normalization does
        occur, an error in the normalization factor will be computed and
        propagated.
    flagmask : {True|False}
        Mask data that is flagged when computing areas of overlap.
    silent : {True|False}
        Print lots of warm fuzzies...

    Returns
    -------
    normspec : muscles spectrum
        normalized spectrum

    Notes
    -----
    - if spectbla has all 0.0 errors (meaning it's a model) normalization
        does not occur
    """

    if not silent:
        names = [s.meta['NAME'] for s in [spectbla, spectblb]]

    # parse out the overlap
    overa, overb = utils.argoverlap(spectbla, spectblb, 'tight')
    if np.sum(overa) == 0 or np.sum(overb) == 0:
        if not silent:
            print ('no full pixel overlap for at least one spectrum. can\'t '
                   'normalize.')
        return 1.0, np.nan

    # if speca has all zero errors (it's a model), don't normalize to it
    if safe and np.all(spectbla[overa]['error'] == 0.0):
        if not silent:
            print ('the master spectrum {} has all zero errors, so {} will '
                   'not be normalized to it'.format(*names))
            return 1.0, np.nan
        #        return spectblb

    # rebin to the coarser spectrum
    if np.sum(overa) < np.sum(overb):
        ospeca = spectbla[overa]
        wbins = utils.wbins(ospeca)
        ospecb = utils.rebin(spectblb, wbins)
        order = slice(None, None, -1)
    else:
        ospecb = spectblb[overb]
        wbins = utils.wbins(ospecb)
        ospeca = utils.rebin(spectbla, wbins)
        order = slice(None, None, 1)
    if not silent:
        over_range = [ospeca['w0'][0], ospeca['w1'][-1]]
        print ('spectra overlap at {:.2f}-{:.2f}'.format(*over_range))
        print ('rebinning {} to the (coarser) resolution of {} where they '
               'overlap'.format(*names[order]))

    # mask data with flags
    if flagmask:
        flagged = (ospeca['flags'] > 0) | (ospecb['flags'] > 0)
        if not silent:
            percent_flagged = np.sum(flagged) / float(len(ospeca)) * 100.0
            print ('{:.2f}% of the data that was flagged in one spectra or '
                   'the other. masking it out.'.format(percent_flagged))
    else:
        flagged = np.zeros(len(ospeca), bool)

    # mask data where speca has 0.0 errors
    if safe:
        zeroerr = (ospeca['error'] == 0.0)
        if not silent:
            percent_zeroerr = np.sum(zeroerr) / float(len(ospeca)) * 100.0
            print ('{:.2f}% of the data in the master spectrum had zero error. '
                   'masking it out'.format(percent_zeroerr))
    else:
        zeroerr = np.zeros(len(ospeca), bool)

    # don't mask out data with low SN because then you bias the result to high flux values

    # compute normalization factor
    ospecs = [ospeca, ospecb]
    dw = wbins[:, 1] - wbins[:, 0]

    good = ~(flagged | zeroerr)
    if np.sum(good) == 0:
        return 1.0, np.nan

    def getarea(spec):
        area = np.sum(spec['flux'][good] * dw[good])
        error = mnp.quadsum(spec['error'][good] * dw[good])
        return area, error

    areas, errors = zip(*map(getarea, ospecs))
    if not silent:
        print ('master spectrum has overlap area    {:.2e} ({:.2e})\n'
               'secondary spectrum has overlap area {:.2e} ({:.2e})'
               ''.format(areas[0], errors[0], areas[1], errors[1]))
    diff = abs(areas[1] - areas[0])
    differr = mnp.quadsum(errors)
    p = 2.0 * (1.0 - norm.cdf(diff, loc=0.0, scale=differr))
    if not silent:
        print ('difference =                        {:.2e} ({:.2e})'
               ''.format(diff, differr))
        print 'probability that the difference is spurious = {:.4f}'.format(p)
    if worry and p > worry:
        if not silent:
            print ('{} > {}, so secondary will not be normalized to master'
                   ''.format(p, worry))
        return 1.0, np.nan

    # area ratio
    normfac = areas[0] / areas[1]

    normfacerr = sqrt((errors[0] / areas[1]) ** 2 + (areas[0] * errors[1] / areas[1] ** 2) ** 2)

    if not silent:
        print ('secondary will be normalized by a factor of {} ({})'
               ''.format(normfac, normfacerr))

    return normfac, normfacerr


def norm2photometry(spec, photom_tbl=None, band_dict=None, silent=False, plotfit=False, return_ln_like=False,
                    return_tbl_and_err=False, clean=False, err='constSN'):

    if not silent: print "Normalizing {} to photometry.".format(spec.meta['NAME'])
    if photom_tbl is None:
        star = spec.meta["STAR"]
        photometry = io.get_photometry(star, spec['w0'][0], spec['w1'][-1], silent=silent)
        if photometry is None:
            return None
        else:
            tbl, band_dict = photometry
    else:
        tbl = photom_tbl

    # convert flux to per freq.
    spec = utils.add_frequency(spec)
    v = (spec['v0'] + spec['v1'])/2.0
    fnu = spec['flux_jy']

    # compute synthetic phot in all bands used in table
    synphot_dict = {}
    for key, band in band_dict.items():
        wb, rb = band.T
        vb = (const.c/(wb*u.AA)).to(u.Hz).value
        rbi = np.interp(v[::-1], vb[::-1], rb[::-1])[::-1]
        synphot_dict[key] = np.trapz(rbi*fnu, v)/np.trapz(rb, vb) # Jy

    if type(err) is not str:
        std = err
    synphot = np.array([synphot_dict[key] for key in tbl['sed_filter']])
    phot = np.array(tbl['sed_flux'], 'f8')
    vp = np.array(tbl['sed_freq'], 'f8')
    cut_prob_all_pts = rc.norm2phot_outlier_cut
    outliers = tbl[0:0]
    while True:
        if len(phot) < 3:
            raise ValueError('Only three photometry points (some may have been clipped). Need at least three to '
                             'safely estimate the scatter.')
        if err == 'constSN':
            S = np.sum(synphot/phot) # for variance ~ flux
            S2 = np.sum(synphot**2/phot**2)
        elif err == 'sqrt':
            S = np.sum(synphot)
            S2 = np.sum(synphot**2/phot)
        else:
            S = np.sum(synphot**2/std**2)
            S2 = np.sum(synphot*phot/std**2)
        normfac = S/S2
        normsynphot = normfac*synphot
        if err == 'constSN':
            const2 = np.sum((normsynphot - phot)**2/normsynphot**2)/2/len(phot)
            std = np.sqrt(const2)*normsynphot
        elif err == 'sqrt':
            const2 = np.sum((normsynphot - phot)**2/normsynphot)/2/len(phot)
            std = np.sqrt(const2*normsynphot)
        normed_residuals = (phot - normsynphot)/std
        if clean:
            cut_prob_single_pt = 1 - (1 - cut_prob_all_pts)**(1.0/len(phot))
            cut_std = pdfutils.inv_gauss_cdf(cut_prob_single_pt)
            keep = abs(normed_residuals) < cut_std
            if np.all(keep):
                break
            vp, phot, synphot = vp[keep], phot[keep], synphot[keep]
            outliers = vstack([outliers, tbl[~keep]])
            tbl = tbl[keep]
        else:
            break

    if return_tbl_and_err:
        return tbl, std

    normfac_err = 1.0/np.sqrt(np.sum((synphot/std)**2))
    synphot, synphot_err = synphot*normfac, synphot*normfac_err
    if not silent:
        print "spectrum normalized by {:.2e} with a {:.1f}% uncertainty".format(normfac, 100*normfac_err/normfac)

    if plotfit not in [None, False]:
        if type(plotfit) is not bool:
            ax0, ax1 = plotfit
        else:
            fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
            ax0.set_position([0.1, 0.3, 0.85, 0.65])
            ax1.set_position([0.1, 0.1, 0.85, 0.2])

        # fit
        ## use mean filter wavelength
        wp = []
        for line in tbl:
            band = band_dict[line['sed_filter']]
            w, T = band.T
            wp.append(np.sum(w*T)/np.sum(T))
        # wp = (const.c/vp).to(u.AA).value

        xlim = [0.9*min(wp), 1.1*max(wp)]
        w = (spec['w0'] + spec['w1'])/2.0
        keep = mnp.inranges(w, xlim)
        w, fnu = w[keep], fnu[keep]
        N = len(fnu)/1000
        if N > 1:
            fnu = mnp.smooth(fnu, N, safe=False)[::N]
            w = mnp.smooth(w, N, safe=False)[::N]
        ax0.plot(w, fnu*normfac, '-k')
        ax0.fill_between(w, fnu*(normfac - normfac_err), fnu*(normfac + normfac_err), edgecolor='none', color='k',
                         alpha=0.3)
        ax0.plot((const.c/outliers['sed_freq']).to(u.AA).value, outliers['sed_flux'], 'rx')
        ax0.errorbar(wp, phot, std, fmt='.', color='g', capsize=0)
        ax0.plot(wp, synphot, 'k+', ms=10)
        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax0.autoscale(axis='y', tight=True)
        ylim = ax0.get_ylim()
        ax0.set_ylim(ylim[0]/1.5, ylim[1]*1.5)
        ax0.set_xlim(xlim)
        ax1.set_xlabel('Wavelength [$\AA$]')
        ax0.set_ylabel('Flux [Jy]')

        # residuals
        ax1.plot(wp, normed_residuals, 'k.')
        ax1.grid()
        ax1.axhline(0.0, color='k')
        ax1.set_ylabel('(O-C)/$\sigma$')
        ax1.set_xscale('log')
        ax1.set_xlim(xlim)

        plt.draw()

    if return_ln_like:
        return -np.sum(0.5*np.log(2*np.pi) + np.log2(std) + 0.5*normed_residuals**2)
    else:
        return normfac, normfac_err


def smartsplice(spectbla, spectblb, minsplice=0.005, silent=False):
    """
    Splice one spectrum into another (presumably overlapping) spectrum in a
    way that minimizes overall error.

    Chooses a contiguous region of the overlap between spetrum a and spectrum
    b such that the integrated error in the overlap is as low as possible.
    This could mean not splicing spectrum b in at all.

    Parameters
    ----------
    spectbla : astropy Table
        Table of the base spectrum. Must not contain any gaps.
    spectblb : astropy Table
        Table of the spectrum to be splice into spectbla.
    minsplice : float
        Minimum size of the splice relative to the central wavelength value
        of the overlapping range. Set to 0.0 in order to allow splices of any
        size. This is only valid for splices where the range of one spectrum
        is entirely within another.

    Returns
    -------
    splicedspec : astropy Table
        The spliced spectrum.
    """

    # if there are zeros or nans for errors in one spectrum but not the other, delete portions as appropriate
    def groom(speca, specb, safe):
        getgood = lambda spec: np.isfinite(spec['error']) & (spec['error'] > 0)
        bada = ~getgood(speca)
        badrangesa = specutils.flags2ranges(utils.wbins(speca), bada)
        arange = utils.gapless_ranges(speca)
        if not np.any(badrangesa):
            return speca
        elif safe:
            goodb = getgood(specb)
            goodrangesb = specutils.flags2ranges(utils.wbins(specb), goodb)
            cutranges = mnp.range_intersect(badrangesa, goodrangesb)
            if len(cutranges) == 0:
                return speca
            keepranges = mnp.rangeset_subtract(arange, cutranges)
            return utils.keepranges(speca, keepranges, ends='exact')
        else:
            brange = utils.gapless_ranges(specb)
            cutranges = mnp.range_intersect(brange, badrangesa)
            if len(cutranges) == 0:
                return speca
            keepranges = mnp.rangeset_subtract(arange, cutranges)
            return utils.keepranges(speca, keepranges, ends='exact')
    spectbla = groom(spectbla, spectblb, safe=True)
    spectblb = groom(spectblb, spectbla, safe=False)
    assert not (len(spectbla) == 0 and len(spectblb) == 0)
    if len(spectbla) == 0:
        return spectblb
    if len(spectblb) == 0:
        return spectbla

    # sort the two spectra
    both = [spectbla, spectblb]
    key = lambda s: s['w0'][0]
    both.sort(key=key)
    spec0, spec1 = both
    if not silent:
        names = [s.meta['NAME'] for s in both]

    if not utils.overlapping(*both):  # they don't overlap
        specs = sum(map(utils.gapsplit, both), [])
        specs.sort(key=key)
        spec = reduce(splice, specs)
        assert np.all(spec['w0'][1:] > spec['w0'][:-1])
        return spec

    # get their overlap and the range of the overlap
    over0, over1 = utils.argoverlap(*both, method='loose')
    ospec0, ospec1 = spec0[over0], spec1[over1]
    oboth = [ospec0, ospec1]
    wr0, wr1 = [[s['w0'][0], s['w1'][-1]] for s in oboth]
    wr = [max(wr0[0], wr1[0]), min(wr0[1], wr1[1])]

    # if the spectra have gaps within the overlap, split them at their gaps,
    # sort them, and splice in pairs
    gaps0, gaps1 = map(utils.gapranges, both)
    gapsin0 = np.any(mnp.inranges(gaps0.flatten(), wr1))
    gapsin1 = np.any(mnp.inranges(gaps1.flatten(), wr0))
    if gapsin0 or gapsin1:
        specsa, specsb = map(utils.gapsplit, both)
        specsa.sort(key=key)
        specsb.sort(key=key)
        spec = specsa.pop(0)
        while len(specsa) > 0:
            ospecs = filter(lambda s: utils.overlapping(spec, s), specsb)
            while len(ospecs):
                spec = smartsplice(spec, ospecs.pop(0))
            spec = smartsplice(specsa.pop(0), spec)
        spec = reduce(splice, specsb, spec)
        assert np.all(spec['w0'][1:] > spec['w0'][:-1])
        return spec

    # otherwise, find the best splice locations
    # get all edges within the overlap
    we0, we1 = [__edgesinrange(s, wr) for s in both]
    we = np.hstack([wr[0], we0, we1])
    we = np.unique(we)  # also sorts the array
    wbins = utils.edges2bins(we)
    # wr[1] is already included because of how searchsorted works

    # rebin spectral overlap to we
    oboth = [utils.rebin(o, wbins) for o in oboth]
    dw = np.diff(we)

    # get flux and variance and mask values with dq flags and nan values
    masks = [(spec['flags'] > 0) for spec in oboth]
    flus = [spec['flux'] * dw for spec in oboth]
    sig2s = [(spec['error'] * dw) ** 2 for spec in oboth]

    def maskitfillit(x, mask):
        x[mask] = 0.0
        x[np.isnan(x)] = 0.0
        return x

    flus = map(maskitfillit, flus, masks)
    sig2s = map(maskitfillit, sig2s, masks)

    sumstuff = lambda x: np.insert(np.cumsum(x), 0, 0.0)
    cf0, cf1 = map(sumstuff, flus)
    cv0, cv1 = map(sumstuff, sig2s)
    # this way, any portions where all variances are zero will result in a
    # total on nan for the signal to noise

    enclosed = (wr1[1] < wr0[1])
    if enclosed:
        wmid = (wr[1] + wr[0]) / 2.0
        mindw = minsplice * wmid
        if len(we) < 500:
            indices = np.arange(len(we))
        else:
            indices = np.round(np.linspace(0, len(we) - 1, 500)).astype(int)
        ijs = combos(indices, 2)
        i, j = np.array([ij for ij in ijs]).T
        keep = (we[j] - we[i]) > mindw
        i, j = i[keep], j[keep]
        i, j = map(np.append, [i, j], [0, 0])

        #        signal = cf0[i] + (cf1[j] - cf1[i]) + (cf0[-1] - cf0[j])
        var = cv0[i] + (cv1[j] - cv1[i]) + (cv0[-1] - cv0[j])
        #        SN = signal/np.sqrt(var)
        # pick the best and splice the spectra
        #        best = np.nanargmax(SN)
        best = np.nanargmin(var)
        i, j = i[best], j[best]
        cut0, cut1 = we[i], we[j]
        if i == j:
            return spec0

        splicespec = spec1
        splicespec = utils.split_exact(splicespec, cut0, 'red')
        splicespec = utils.split_exact(splicespec, cut1, 'blue')
        if not silent:
            print ('spectrum {} spliced into {} from {:.2f} to {:.2f}'
                   ''.format(names[1], names[0], cut0, cut1))

    # do the same, if not enclosed
    else:
        i = range(len(we))
        #        signal = cf0[i] + (cf1[-1] - cf1[i])
        var = cv0[i] + (cv1[-1] - cv1[i])
        #        SN = signal/np.sqrt(var)
        #        best = np.nanargmax(SN)
        best = np.nanargmin(var)
        i = i[best]
        cut = we[best]
        splicespec = utils.split_exact(spec1, cut, 'red')

        if not silent:
            print ('spectrum {} spliced into {} from {:.2f} onward'
                   ''.format(names[1], names[0], cut))

    spec = splice(spec0, splicespec)
    assert np.all(spec['w0'][1:] > spec['w0'][:-1])
    return spec


def splice(spectbla, spectblb):
    """
    Replace spectrum a with spectrum b where they overlap.

    The outer bins of spectrum b are preserved, whereas the bins adjacent
    to the edges of spectrum b in spectrum a may be cut off. If so, the errors
    for the fractional bins are appropriately augmented assuming Poisson
    statistics and a constant flux within the original bins.
    """
    # if spectrum b has gaps, divide it up and add the pieces it into spectbla
    # separately

    if len(spectbla) == 0:
        return spectblb
    if len(spectblb) == 0:
        return spectbla

    if utils.hasgaps(spectblb):
        bspecs = utils.gapsplit(spectblb)
        return reduce(splice, bspecs, spectbla)

    # cut up a according to the range of b and stack
    speclist = []
    leftspec = utils.split_exact(spectbla, spectblb['w0'][0], 'blue')
    speclist.append(leftspec)
    speclist.append(spectblb)
    rightspec = utils.split_exact(spectbla, spectblb['w1'][-1], 'red')
    speclist.append(rightspec)
    spec = utils.vstack(speclist)

    # modify metadata
    metas = [s.meta for s in [spectbla, spectblb]]

    def parsesources(meta):
        if len(meta['SOURCESPECS']):
            return meta['SOURCESPECS']
        else:
            return meta['NAME']

    sources = np.hstack(map(parsesources, metas))
    spec.meta['SOURCESPECS'] = np.unique(sources)
    spec.meta['FILENAME'] = ''
    spec.meta['NAME'] = 'stitched spectrum'

    assert np.all(spec['w0'][1:] > spec['w0'][:-1])
    return spec


def coadd(spectbls, maskbaddata=True, savefits=False, weights='exptime',exptime='sum',  silent=False):
    """Coadd spectra in spectbls. weights can be 'exptime' or 'error'"""
    inst = __same_instrument(spectbls)
    # star = __same_star(spectbls)
    star = spectbls[0].meta['STAR']

    # split spectra at gaps to avoid removing the gaps
    temp = map(utils.gapsplit, spectbls)
    spectbls = sum(temp, [])

    sourcefiles = [s.meta['FILENAME'] for s in spectbls]

    listify = lambda s: [spec[s].data for spec in spectbls]
    w0, w1, f, e, expt, dq, inst, normfac, start, end = map(listify, colnames)
    we = [np.append(ww0, ww1[-1]) for ww0, ww1 in zip(w0, w1)]

    if any([np.any(n != 1.0) for n in normfac]):
        warn("Spectra with normfacs != 1.0 are being coadded.")

    weights = [1.0 / ee ** 2 for ee in e] if weights == 'error' else expt
    if maskbaddata:
        dqmasks = map(rc.seriousdqs, sourcefiles)
        masks = __make_masks(we, dq, dqmasks)
        for i in range(len(masks)):
            start[i][masks[i]] = np.inf
            end[i][masks[i]] = -np.inf
            inst[i][masks[i]] = 0
        cwe, cf, ce, cexpt, dq = specutils.coadd(we, f, e, weights, dq, masks)
    else:
        cwe, cf, ce, cexpt, dq = specutils.coadd(we, f, e, weights, dq)


    data = [inst, start, end]
    funcs = ['or', 'min', 'max']
    basevals = [0, np.inf, -np.inf]

    def specialcoadder(a, f, bv):
        return specutils.stack_special(we, a, f, commongrid=cwe, baseval=bv)

    cinst, cstart, cend = map(specialcoadder, data, funcs, basevals)
    cnorm = np.ones(len(cwe) - 1)
    cw0, cw1 = cwe[:-1], cwe[1:]
    if exptime == 'pass':
        cexpt = specialcoadder(expt, 'max', 0.0)

    goodbins = (cexpt > 0)
    data = [v[goodbins] for v in [cw0, cw1, cf, ce, cexpt, dq, cinst, cnorm, cstart, cend]]
    cfile = db.coaddpath(sourcefiles[0])
    cname = db.parse_name(cfile)
    sourcespecs = list(set([s.meta['NAME'] for s in spectbls]))
    spectbl = utils.list2spectbl(data, star, name=cname, sourcespecs=sourcespecs)

    if all([np.all(s['instrument'] > 0) for s in spectbls]):
        assert np.all(spectbl['instrument'] > 0)
    assert not np.any(spectbl['minobsdate'] < 0)
    if np.all(spectbl['minobsdate'] > 0):
        assert np.all(spectbl['maxobsdate'] > spectbl['minobsdate'])
    else:
        assert np.all(spectbl['maxobsdate'] >= spectbl['minobsdate'])

    if savefits:
        if type(savefits) is str: cfile = savefits
        io.writefits(spectbl, cfile, overwrite=True)
        spectbl.meta['FILENAME'] = cfile
        if not silent: print 'coadd saved to \n\t{}'.format(cfile)
    return spectbl


def auto_coadd(star, configs=None, silent=False):
    if configs is None:
        groups = db.coaddgroups(star)
    else:
        if type(configs) is str: configs = [configs]
        groups = []
        for config in configs:
            files = db.sourcespecfiles(star, config)
            files = db.sub_customfiles(files)
            # clooge to deal with case when there are no x1ds -- look for just custom specs
            if len(files) == 0:
                files = db.findfiles('u', star, config, 'custom_spec', fullpaths=True)
            groups.append(files)

    for group in groups:
        if not silent:
            names = [os.path.basename(f) for f in group]
            print 'coadding the spectra \n\t{}'.format('\n\t'.join(names))
        echelles = map(utils.isechelle, group)
        if any(echelles):
            if not silent:
                print 'some files are echelles, so weighting orders by 1/error**2 and then spectra by exptime'
            spectbls = []
            for f in group:
                orders = io.read(f)
                spec = coadd(orders, savefits=False, weights='error', exptime='pass', silent=silent)
                spec.meta['FILENAME'] = f
                spec.meta['NAME'] = orders[0].meta['NAME']
                spectbls.append(spec)
        else:
            spectbls = io.read(group)
        if len(spectbls) == 1:
            if any(echelles):
                path = db.coaddpath(group[0])
                if not silent:
                    print 'single echelle spectrum, saving to {}'.format(path)
                io.writefits(spectbls[0], path, overwrite=True)
            else:
                if not silent:
                    print 'single spectrum for {}, moving on'.format(spectbls[0].meta['NAME'])
            continue
        else:
            if not silent:
                print 'weighting by exposure time and coadding'
            coadd(spectbls, savefits=True, weights='exptime', exptime='sum', silent=silent)


def phxspec(Teff, logg=4.5, FeH=0.0, aM=0.0, repo=rc.phxrepo):
    """
    Quad-linearly interpolates the available phoenix spectra to the provided
    values for temperature, surface gravity, metallicity, and alpha metal
    content.
    """
    grids = [rc.phxTgrid, rc.phxggrid, rc.phxZgrid, rc.phxagrid]
    pt = [Teff, logg, FeH, aM]

    # make a function to retrieve spectrum given grid indices
    def getspec(*indices):
        args = [grid[i] for grid, i in zip(grids, indices)]
        return io.phxdata(*args, repo=repo)

    # interpolate
    spec = mnp.sliminterpN(pt, grids, getspec)

    # make spectbl
    N = len(spec)
    err = np.zeros(N)
    expt, flags = np.zeros(N), np.zeros(N, 'i1')
    insti = rc.getinsti('mod_phx_-----')
    source = insti * np.ones(N, 'i1')
    normfac, start, end = 1.0, 0.0, 0.0
    data = [rc.phxwave[:-1], rc.phxwave[1:], spec, err, expt, flags, source,
            normfac, start, end]
    return utils.list2spectbl(data)


def auto_phxspec(star, Teff='oldfit', silent=False, err='constSN', fitspec=None):
    kwds = {}
    for key in ['logg', 'FeH', 'aM']:
        val = rc.starprops[key][star]
        if not np.isnan(val):
            kwds[key] = val
    Tlit = rc.starprops['Teff'][star]
    Tlit_err = rc.starprops.errpos['Teff'][star]

    if Teff == 'fit' and fitspec is None:
        # load in photometry data
        tbl, band_dict = io.get_photometry(star, rc.vis[0], 55000.0)

        # find minimum using the adaptive error bars and culling outliers
        def ln_like(Teff):
            spec = phxspec(Teff, **kwds)
            return norm2photometry(spec, photom_tbl=tbl, band_dict=band_dict, silent=True, plotfit=False,
                                   return_ln_like=True, clean=True, err=err)
        if not silent: "Finding Teff that best fit photometry."
        result = opt.minimize_scalar(lambda Teff: -ln_like(Teff), bracket=[Tlit-Tlit_err, Tlit+Tlit_err],
                                     bounds=[Tlit-500, Tlit+500], method='bounded', options={'disp': (not silent),
                                                                                             'xtol': 10.})
        Teff = result.x
        spec = phxspec(Teff, **kwds)
        if not silent: print "Best fit Teff of {:.0f} found. Finding confidence interval.".format(Teff)

        # cull outliers for optimal solution, then find error bars using error bars and points from optimal solution
        tbl, uncts = norm2photometry(spec, photom_tbl=tbl, band_dict=band_dict, silent=True, plotfit=False,
                                   return_tbl_and_err=True, clean=True, err=err)
        def constrained_like(Teff):
            spec = phxspec(Teff, **kwds)
            return norm2photometry(spec, photom_tbl=tbl, band_dict=band_dict, silent=True, plotfit=False,
                                   return_ln_like=True, err=uncts)
        N = 201
        for dT in range(100,2001,100):
            try:
                # remember result.fun is negative log like
                Tlim = opt.brentq(lambda T: (result.fun + 2) + constrained_like(T), Teff, Teff+dT)
                break
            except ValueError:
                continue
        dT = Tlim - Teff
        Tlo = max(2300, Teff - dT)
        Tgrid = np.linspace(Tlo, Teff+dT, N)
        Lgrid = np.exp(map(ln_like, Tgrid))
        Igrid = np.cumsum(np.diff(Tgrid)*mnp.midpts(Lgrid))
        Imid = Igrid[N/2]
        I = Igrid[-1]
        dI = I*0.683/2
        I0, I1 = Imid - dI, Imid + dI
        T0, T1 = np.interp([I0, I1], Igrid, mnp.midpts(Tgrid))
        if not silent: print ("Best-fit phoenix spectrum found with Teff = {:.0f} ({:.0f}-{:.0f}) comapred to "
                              "literature value of {:.0f}. Saving.".format(Teff, T0, T1, Tlit))
    if Teff == 'fit' and fitspec is not None:

        buffer = 100.
        dw = np.mean(fitspec['w1'] - fitspec['w0'])
        fitspec = utils.evenbin(fitspec, dw)
        bins = utils.bins2edges(utils.wbins(fitspec))
        bins_end = np.arange(dw, buffer+dw, dw) + bins[-1]
        bins_beg = np.arange(-buffer, 0, dw) + bins[0]
        bins_buffered = np.hstack([bins_beg, bins, bins_end])
        bins_buffered = utils.edges2bins(bins_buffered)

        def ln_like(Teff):
            spec = phxspec(Teff, **kwds)

            # trim with some buffer
            spec = utils.rebin(spec, bins_buffered)

            # roughly normalize
            fac, _ = normalize(fitspec, spec, safe=False, silent=True)
            spec['flux'] *= fac

            # align wavelengths and cut out appropriate piece of spec
            offset = mnp.align(spec['flux'], fitspec['flux'])
            assert offset >= 0
            assert offset < len(spec) - len(fitspec)
            i0, i1 = offset, offset + len(fitspec)
            spec = spec[i0:i1]

            # normalize again, more precise now that specs are aligned
            fac, _ = normalize(fitspec, spec, safe=False, silent=True)
            spec['flux'] *= fac

            # compute ln like
            terms = -(spec['flux'] - fitspec['flux'])**2/2/fitspec['err']**2
            return np.sum(-np.log(np.sqrt(2*np.pi)*fitspec['err']) + terms)

        if not silent: print "Finding Teff that best fit the provided spectrum."
        result = opt.minimize_scalar(lambda Teff: -ln_like(Teff), bracket=[Tlit-Tlit_err, Tlit+Tlit_err],
                                     bounds=[Tlit-500, Tlit+500], method='bounded', options={'disp': (not silent),
                                                                                             'xtol': 10.})
        Teff = result.x
        spec = phxspec(Teff, **kwds)
        if not silent: print "Best fit Teff of {:.0f} found. Finding confidence interval.".format(Teff)

        # cull outliers for optimal solution, then find error bars using error bars and points from optimal solution
        tbl, uncts = norm2photometry(spec, photom_tbl=tbl, band_dict=band_dict, silent=True, plotfit=False,
                                   return_tbl_and_err=True, clean=True, err=err)
        N = 201
        for dT in range(100,2001,100):
            try:
                # remember result.fun is negative log like
                Tlim = opt.brentq(lambda T: (result.fun + 2) + ln_like(T), Teff, Teff+dT)
                break
            except ValueError:
                continue
        dT = Tlim - Teff
        Tlo = max(2300, Teff - dT)
        Tgrid = np.linspace(Tlo, Teff+dT, N)
        Lgrid = np.exp(map(ln_like, Tgrid))
        Igrid = np.cumsum(np.diff(Tgrid)*mnp.midpts(Lgrid))
        Imid = Igrid[N/2]
        I = Igrid[-1]
        dI = I*0.683/2
        I0, I1 = Imid - dI, Imid + dI
        T0, T1 = np.interp([I0, I1], Igrid, mnp.midpts(Tgrid))
        if not silent: print ("Best-fit phoenix spectrum found with Teff = {:.0f} ({:.0f}-{:.0f}) comapred to "
                              "literature value of {:.0f}. Saving.".format(Teff, T0, T1, Tlit))

    if Teff == 'lit':
        Teff = Tlit
        T0 = Tlit - Tlit_err
        T1 = Tlit + Tlit_err
    if Teff == 'oldfit':
        Teff = rc.starprops['Teff_muscles'][star]
        T0 = Teff - rc.starprops.errneg['Teff_muscles'][star]
        T1 = Teff + rc.starprops.errpos['Teff_muscles'][star]
    if not silent:
        print 'interpolating phoenix spectrum for {} with values'.format(star)
        print "Teff = {}".format(Teff), kwds
    spec = phxspec(Teff, **kwds)
    spec.meta['STAR'] = star
    spec.meta['Teff'] = Teff
    spec.meta['Terrneg'] = Teff - T0 if Teff in ['fit', 'oldfit'] else np.nan
    spec.meta['Terrpos'] = T1 - Teff if Teff in ['fit', 'oldfit'] else np.nan
    path = rc.phxpath(star)
    spec.meta['NAME'] = db.parse_name(path)
    if not silent:
        print 'writing spectrum to {}'.format(path)
    io.writefits(spec, path, overwrite=True)

    return spec


def auto_customspec(star, silent=False):
    ss = rc.loadsettings(star)
    if not silent:
        if len(ss.custom_extractions) == 0:
            'no custom extractions set for {}'.format(star)
    for custom in ss.custom_extractions:
        config = custom['config']
        if not silent:
            print 'custom extracting spectrum for {}'.format(config)
            print 'with parameters'
            print custom['kwds']
        if 'hst' in config:
            x2dfiles = db.findfiles('u', star, config, 'x2d', fullpaths=True)
            for x2dfile in x2dfiles:
                if not silent:
                    print 'using x2dfile {}'.format(x2dfile)
                x1dfile = x2dfile.replace('x2d', 'x1d')
                if not os.path.exists(x1dfile): x1dfile = None
                specfile = x2dfile.replace('x2d', 'custom_spec')
                dqmask = rc.seriousdqs(specfile)
                spec = x2dspec(x2dfile, x1dfile=x1dfile, bkmask=dqmask, **custom['kwds'])

                # trim any nans
                isnan = np.isnan(spec['flux'])
                spec = spec[~isnan]

                # conform to spectbl standard
                meta = spec.meta
                datalist = [spec[s] for s in ['w0', 'w1', 'flux', 'error']]
                hdr = fits.getheader(x2dfile, extname='sci')
                expt, start, end = [hdr[s] for s in ['exptime', 'expstart', 'expend']]
                inst = db.getinsti(specfile)
                norm = 1.0
                datalist.extend([expt, spec['dq'], inst, norm, start, end])

                spectbl = utils.list2spectbl(datalist, star, specfile, '', [os.path.basename(x2dfile)])
                for key in meta:
                    spectbl.meta[key] = meta[key]

                if not silent:
                    print 'saving custom extraction to {}'.format(specfile)
                io.writefits(spectbl, specfile, overwrite=True)
        else:
            raise NotImplementedError("No custom extractions defined for {}"
                                      "".format(config))


def auto_photons(star, inst='all', fluxed='tag_vs_x1d'):
    alltagfiles = db.findfiles('u', 'tag', star, fullpaths=True)
    # sets = rc.loadsettings(star)

    if inst == 'all':
        instruments = map(db.parse_instrument, alltagfiles)
        instruments = list(set(instruments))
        inst = filter(lambda s: 'cos' in s, instruments)

    if type(inst) == list:
        [auto_photons(star, i) for i in instruments]

    getInstFiles = lambda files: filter(lambda s: inst in s, files)
    tagfiles = getInstFiles(alltagfiles)
    if len(tagfiles) == 0:
        print 'No tag files found for the {} instrument.'.format(inst)
        return
    x1dfiles = [db.findsimilar(tf, 'x1d')[0] for tf in tagfiles]

    # kwds = sets.get_tag_extraction(inst)
    # if kwds is None:
    #     if 'cos_g230l' in inst:
    #         kwds = {'extrsize':30, 'bkoff':[30, -30], 'bksize':[20, 20]}
    #     else:
    #         kwds = {}

    photons_list = [sp.hst.readtag(tf, xf, fluxed=fluxed) for tf, xf in zip(tagfiles, x1dfiles)]
    if any(['corrtag_b' in tf for tf in tagfiles]):
        photons_a = filter(lambda p: p.obs_metadata[0]['segment'] == 'FUVA', photons_list)
        photons_b = filter(lambda p: p.obs_metadata[0]['segment'] == 'FUVB', photons_list)
        photons_a, photons_b = [sum(ps[1:], ps[0]) for ps in [photons_a, photons_b]]
        pfa, pfb = [db.photonpath(star, inst, s) for s in ['a', 'b']]
        [p.writeFITS(pf, overwrite=True) for p,pf in [[photons_a, pfa], [photons_b, pfb]]]
    else:
        seg = 'a' if 'corrtag' in tagfiles[0] else ''
        pf = db.photonpath(star, inst, seg)
        photons = sum(photons_list[1:], photons_list[0])
        photons.writeFITS(pf, overwrite=True)


def remove_line(spec, wavelength, fill_with=None, minclip=None, silent=False):
    """
    Remove a line from a spectrum by automatically identifying its extent and
    clipping it out.

    Parameters
    ----------
    spec : muscles spectbl
    wavelength : float
        central wavelength of line
    fill_with : {None|int}
        Whether to fill the resulting gap. If None, just return spectrum
        with a gap. If an integer, fit a polynomial of that order and use
        it to fill the gap.
    minclip : float, optional
        Minumum wavlength range to clip out of spectrum relative to the
        line wavelength. E.g. 0.001 means clip out at elast 0.001 * wavelength
        if no line is identified.

    Returns
    -------
    spec : muscles spectbl
        spectrum with line removed or filled, if found
    """
    w = wavelength
    n = 4 if fill_with is None else fill_with

    def fallback_range():
        radius = w / minclip / 2.0
        if not silent:
            print ('using minclip range of +- {} AA (R = {})'
                   ''.format(radius, minclip))
        return [w - radius, w + radius]

    # clip spectrum down to just a window around the line for fast processing
    width = w * 0.01
    window = [w - width / 2.0, w + width / 2.0]
    argmini = utils.argrange(spec, window)

    # exclude any zero-error values. polyfit can't handle 'em
    argmini[spec['error'] <= 0.0] = False
    minispec = spec[argmini]

    # identify lines vs continuum in spectrum
    wbins = utils.wbins(minispec)
    flux, error = minispec['flux'], minispec['error']
    pcont, pline = 1.0 - 1e-5, 1.0 - 1e-10
    try:
        flags = specutils.split(wbins, flux, error, contcut=pcont,
                                linecut=pline, contfit=n, silent=silent)

        # record the continuum points in spec that should be fit
        cont = (flags == 3)
        fit_pts = np.nonzero(argmini)[0]
        fit_pts = fit_pts[cont]

        # isolate range of emission feature that includes wavelength
        line_ranges = specutils.flags2ranges(wbins, flags == 1)
        inrange = (w <= line_ranges[:, 1]) & (w >= line_ranges[:, 0])

        # if an emission feature does include it, get its range
        if np.any(inrange):
            if not silent:
                print 'successfully located line at {} AA'.format(w)
            bad_range = np.squeeze(line_ranges[inrange, :])
        else:
            if not silent:
                print ('none of the identified lines covered the provided '
                       'wavelength of {:.2f} AA'.format(w))
            if minclip is None:
                if not silent:
                    print 'returning unchanged spectrum'
                return spec
            else:
                bad_range = fallback_range()
    except ValueError:
        if not silent:
            print ('could not separate line from continuum emission within '
                   '{:.2f}-{:.2f}'.format(*window))
        if minclip is None:
            if not silent:
                print 'returning unchanged spectrum'
            return spec
        else:
            bad_range = fallback_range()
            fit_pts = argmini

    gap = utils.argrange(spec, bad_range)
    comment = 'clipped out data from {:.2f}-{:.2f} AA'.format(*bad_range)
    if not silent:
        print comment
    spec.meta['COMMENT'].append(comment)

    # fill the gap, if desired
    if fill_with is None:
        spec = spec[~gap]
        if not silent:
            print 'leaving a gap where data was clipped out'
    else:
        # get the bins to be filled
        gapbins = utils.wbins(spec[gap])

        # fill 'em
        spec = fill_gaps(spec, n, fit_pts=fit_pts, gapbins=gapbins,
                         silent=silent)

    return spec


def fill_gaps(spec, fill_with=4, fit_span=10.0, fit_pts=None, resolution=None,
              mingapR=10.0, gapbins=None, silent=False):
    """
    Fill gaps in the spectrum with a polynomial fit to the continuum in an
    area encompassing the gap.

    Parameters
    ----------
    spec : muscles spectbl
    fill_with : int, optional
        Order of the polynomial fit to use for filling the gap
    fit_span : float, optional
        Width of the area centered on the gap within which to fit the
        polynomial, in multiples of the gap width.
    fit_pts : 1D array
        Boolean array the same length as spec that specifies which points
        to use for fitting the polynomial. Overrides fit_span.
    resolution : float, optional
        The resolution at which to bin the polynomial for filling the gap. If
        None, the average resolution in fit_span or fit_pts is used.
    gapbins : 2D array
        Specific bins to use when filling a single known gap. Overrides
        resolution. The bins are not checked to see if they are flush with the
        gap borders.
    silent : {True|False}, optional

    Returns
    -------
    filledspec : muscles spectbl
        Spectrum with the gap filled.
    """
    findgaps = (gapbins is None)
    if findgaps and not utils.hasgaps(spec):
        if not silent:
            print 'No gaps in spectrum.'
        return spec

    wbins = utils.wbins(spec)
    flux, error = spec['flux'], spec['error']
    n = fill_with

    # if fit_pts provided, go ahead and fit polynomial to those
    if fit_pts is not None:
        poly = specutils.polyfit(wbins[fit_pts, :], flux[fit_pts], n,
                                 error[fit_pts])[2]

    # identify gaps
    if findgaps:
        gapranges = utils.gapranges(spec)
    else:
        gapranges = np.array([[gapbins[0, 0], gapbins[-1, 1]]])

    gapspecs = []
    comments = []
    for gr in gapranges:
        width = gr[1] - gr[0]
        midpt = (gr[0] + gr[1]) / 2.0
        gapR = midpt / width
        if gapR < mingapR:
            if not silent:
                print ('gap from {:.2f}-{:.2f} has R = {:.1f} < {:.1f} = '
                       'mingap, skipping'.format(gr[0], gr[1], gapR, mingapR))
            gapspecs.append(spec[0:0])
            continue
        if fit_pts is None:
            # confine spectrum to just fit_span around gap
            radius = (fit_span * width) / 2.0
            wspan = [midpt - radius, midpt + radius]
            span = utils.argrange(spec, wspan)

            if np.sum(span) < 100:
                i = utils.specwhere(spec, midpt)[1]
                span[i - 50:i + 50] = True

            if not silent:
                print 'fitting polynomial to range {:.2f}-{:.2f}'.format(*wspan)
                print 'attempting to mask out spectral lines'

            # also exclude zero-error (model) points
            span[spec['error'] <= 0.0] = False
            # and points from a different instrument than abuts the gap, unless
            # the gap is between data from two different instruments
            inst1 = spec[spec['w0'] > midpt]['instrument'][0]
            inst0 = spec[spec['w1'] < midpt]['instrument'][-1]
            if inst0 == inst1:
                span[spec['instrument'] != inst0] = False

            # try to filter out emission/absoprtion lines
            try:
                flags = specutils.split(wbins[span], flux[span], error[span],
                                        contcut=1.7, linecut=1.7, contfit=n,
                                        silent=silent)
                cont = (flags == 3)
                wb, f, e = [a[span][cont] for a in [wbins, flux, error]]
                if not silent:
                    pctmasked = 100.0 - 100.0 * np.sum(cont) / np.sum(span)
                    print ('succesfully masked lines including {:.1f}% of the '
                           'data'.format(pctmasked))
            except ValueError:
                if not silent:
                    print 'couldn\'t separate out continuum points'
                wb, f, e = [a[span] for a in [wbins, flux, error]]

            # fit polynomial to data
            poly = specutils.polyfit(wb, f, n, e)[2]
        else:
            wspan = [spec['w0'][fit_pts][0], spec['w1'][fit_pts][-1]]

        if resolution is None and findgaps:
            # compute average resolution in the span around gap
            dw = np.diff(wbins[span], 1)
            resolution = np.mean(dw)

        if findgaps:
            # make a grid to cover the gap
            m = round(width / resolution)
            gridedges = np.linspace(gr[0], gr[1], m + 1)
            gapbins = utils.edges2bins(gridedges)

        if not silent:
            print ('gap from {:.2f}-{:.2f} filled with order {} polynomial'
                   ''.format(gr[0], gr[1], n))

        # compute value of polynomial on the grid
        gapflux = poly(gapbins)[0]

        # add a comment about what will be filled to put in the spectbl meta
        comments.append('{}-{} filled with order {} polynomial fit to the '
                        'continuum within {}-{}'
                        ''.format(gr[0], gr[1], n, wspan[0], wspan[1]))

        # make a spectbl to fill in the gap
        w0, w1 = gapbins.T
        inst = rc.getinsti('mod_gap_fill-')
        star = spec.meta['STAR']
        name = spec.meta['NAME']
        gapspec = utils.vecs2spectbl(w0, w1, gapflux, instrument=inst,
                                     star=star, name=name)
        gapspecs.append(gapspec)

    # stack 'em
    if findgaps:
        # split at all gaps
        dataspecs = utils.gapsplit(spec)
    else:
        # if gapbins provided, just plit the spectrum at the specified gap
        blu = spec[spec['w1'] <= gapranges[0, 0]]
        red = spec[spec['w0'] >= gapranges[0, 1]]
        dataspecs = [blu, red]
    allspecs = [None] * (2 * len(gapspecs) + 1)
    allspecs[0::2] = dataspecs
    allspecs[1::2] = gapspecs
    filledspec = utils.vstack(allspecs, name=spec.meta['NAME'])
    filledspec.meta['COMMENT'].extend(comments)

    return filledspec


# FLARE STUFF
# ===========
def findflares(curveList, flagfactor=1.0, silent=True):
    """
    Return the start and stop time for all excursions (runs) from the median. Flag the excursions that have areas
    flagfactor greater than the mean. Takes a smoothed curve (t0, t1, rate, err), not a regular lightcurve.
    """

    ns = [len(curve[0]) for curve in curveList]
    expends = list(np.cumsum(ns))
    expstarts = [0] + expends[:-1]
    t0s, t1s, rates, errs = map(np.hstack, zip(*curveList))
    ts = (t0s + t1s) / 2.0
    clean = np.ones(len(rates), bool)
    count = 0
    while True:
        if count > 1000:
            break
        normrates = rates - np.median(rates[clean])
        runSliceList, areaList, begList, endList = [], [], [], []
        for i0, i1 in zip(expstarts, expends):
            t, t0, t1, normrate = [a[i0:i1] for a in [ts, t0s, t1s, normrates]]
            runslices = mnp.runslices(normrate)
            tslices = (t[runslices-1] + t[runslices]) / 2.0
            beg = np.insert(tslices, 0.0, t0[0])
            end = np.append(tslices, t1[-1])
            assert np.all((end - beg) > 0)
            begList.append(beg)
            endList.append(end)
            runSliceList.append(runslices + i0)

            # add slice between exposures
            runSliceList.append(i1)

            dt = np.append(np.diff(t0), t1[-1]-t1[-2])
            areas = normrate * dt
            areaList.append(mnp.splitsum(areas, runslices))

        runslices, areas, begs, ends = map(np.hstack, [runSliceList, areaList, begList, endList])
        runslices = np.insert(runslices, 0, 0)

        flare = areas > -(flagfactor * areas.min())
        oldclean = clean
        clean = np.ones(len(rates), bool)
        for i in np.nonzero(flare)[0]:
            i0, i1 = runslices[[i, i+1]]
            clean[i0:i1] = False

        if np.all(clean == oldclean):
            break

    # FIXME: this whole thing should probably go in flare stats, but in particular I need to add error on quiescent flux
    qrate = np.median(rates[clean])
    Fpeaks, FpeakErrs, tpeaks, ratios, ratioerrs = [], [], [], [], []
    for irun in range(len(areas)):
        i0, i1 = runslices[[irun, irun+1]]
        ipeak = np.argmax(np.abs(rates[i0:i1])) + i0
        Fpeaks.append(rates[ipeak] - qrate)
        FpeakErrs.append(errs[ipeak])
        ratios.append(rates[ipeak]/qrate)
        # FIXME: this should account for error in qrate
        ratioerrs.append(errs[ipeak]/qrate)
        tpeaks.append(ts[ipeak])

    return begs, ends, flare, Fpeaks, FpeakErrs, tpeaks, ratios, ratioerrs


def computeFlareStats(fitsphotons, begs, ends, flares, waveranges, dist):
    """
    Compute photometric equivalent widths and absolute energies for the regions defined by begs and ends (in s,
    referenced to MJD0 in fitsphotons) excluding those flagged as flares when computing the mean rate. dist in pc.
    """

    # keep weight and time info for only the photons we want
    photons = fitsphotons['events'].data
    keep = mnp.inranges(photons['wavelength'], waveranges)
    photons = photons[keep]

    # compute mean rate and flux
    cleanranges = np.array([begs[~flares], ends[~flares]])
    ttotal = np.sum(cleanranges[1] - cleanranges[0]) # s
    keep = mnp.inranges(photons['time'], cleanranges)
    mnrate = np.sum(photons[keep]['epsilon']) / ttotal # cnts s-1
    mnflux = np.sum(photons[keep]['epera']) / ttotal # erg s-1 cm-2

    # figure out where time bin edges fit in photons
    i0 = np.searchsorted(photons['time'], begs)
    i1 = np.searchsorted(photons['time'], ends)

    # for each bin, compute PEW
    epssum = np.insert(photons['epsilon'].cumsum(), 0, 0.0)
    assert epssum[-1] < 1e308 # otherwise I shouldn't use cumsum
    totals = epssum[i1] - epssum[i0]
    dts = ends - begs
    rates = totals / dts
    PEWs = (rates - mnrate) / mnrate * dts
    assert np.all(np.isfinite(PEWs))

    # compute ratio to cutoff PEW
    # FIXME: getting some negative PEW ratios, specifically for GJ832 SiIV
    PEWratios = np.abs(PEWs / PEWs.min())

    # multiply by mean luminosity to get absolute flare energy
    dist = dist * 3.08567758e18 # pc to cm
    mnlum = mnflux * 4 * np.pi * dist**2
    Es = PEWs * mnlum

    return PEWs, Es, PEWratios, mnrate, mnflux


def cumfreq(PEWs, exptime):
    """
    Compute cumulative flare frequency given PEWs of detected flares within total exposure time exptime [s].
    """
    isort = np.argsort(PEWs)[::-1]
    cfs = np.zeros(len(PEWs))
    cfs[isort] = (np.arange(len(PEWs)) + 1.0) / (exptime/3600.0/24.0)
    return cfs


def auto_flares(star, bands, inst, label, dt=1.0, silent=False):

    ph, photons = io.readphotons(star, inst)

    nexp = len(ph['gti'].data['obsids'])
    expt = ph[0].header['EXPTIME']

    curves = []
    groups = [range(len(bands))]
    for i in range(nexp):
        ii = (photons['expno'] == i)
        curve = sp.spectral_curves(photons['time'][ii], photons['wavelength'][ii], eps=photons['epsilon'][ii],
                                   tbins=dt, bands=bands, groups=groups)
        tedges, cps, err = zip(*curve)[0]
        t0, t1 = tedges[:-1], tedges[1:]
        curves.append([t0, t1, cps, err])

    begs, ends, flares, Fpeaks, FpeakErrs, tpeaks, ratios, ratioerrs = findflares(curves, silent=silent)

    dist = rc.starprops['dist'][star]
    pews, Es, pewratios, mnrate, mnflux = computeFlareStats(ph, begs, ends, flares, bands, dist)

    assert pews.min() < 0

    cfs = cumfreq(pews, expt)

    data = [begs, ends, tpeaks, pews, flares, cfs, Es, pewratios, Fpeaks, FpeakErrs, ratios, ratioerrs]
    names = ['start', 'stop', 'peak', 'PEW', 'flare', 'cumfreq', 'energy', 'PEWratio', 'Fpk', 'Fpkerr', 'pkratio',
             'pkratioerr']
    units = ['s', 's', 's', 's', '', 'd-1', 'erg', '', 'counts s-1', 'counts s-1', '', '']
    descriptions = ['start time of excursion referenced to MJD0', 'end time of excursion referenced to MJD0',
                    'time of flare peak referenced to MJD0',
                    'photometric equivalent width', 'excursion flagged as a flare',
                    'cumulative frequency of flares of >= PEW', 'absolute energy radiated by flare in band',
                    'ratio of the PEW to the minimum PEW measured in the lightcurve',
                    'peak flux of flare', 'error in peak flux of flare', 'ratio of peak to quiescent flux',
                    'error in ratio of peak to quiescent flux']
    cols = []
    for d, n, u, dc in zip(data, names, units, descriptions):
        cols.append(Table.Column(d, n, unit=u, description=dc))
    flareTable = Table(cols)
    flareTable.meta['STAR'] = star
    flareTable.meta['EXPTIME'] = expt
    flareTable.meta['MJD0'] = ph[0].header['MJD0']
    flareTable.meta['QSCTCPS'] = mnrate
    flareTable.meta['QSCTFLUX'] = mnflux
    flareTable.meta['DT'] = dt

    for i, (w0, w1) in enumerate(bands):
        key0, key1 = 'bandbeg{}'.format(i), 'bandend{}'.format(i)
        flareTable.meta[key0] = w0
        flareTable.meta[key1] = w1

    flareTable.write(db.flarepath(star, inst, label), format='fits', overwrite=True)


def match_flares(star, inst, bandlabels='all', masterband='broad130a', flarecut=None):
    """
    Match the flares from the tables of flares for the bands sepcified by bandlables for specified star and instrument.
    Compute stats.
    """
    if bandlabels == 'all':
        fs = db.findfiles(rc.flaredir, inst, star, 'flares', fullpaths=True)
        bandlabels = [db.parse_info(f, 4, 5) for f in fs]
    try:
        bandlabels.remove(masterband)
    except ValueError:
        pass

    masterCat, _ = io.readFlareTbl(star, inst, masterband)

    colnames = masterCat.colnames
    masternames = [masterband + ' ' + name for name in colnames]
    matchCat = Table(data=masterCat, names=masternames)

    for band in bandlabels:
        for name in colnames:
            newname = band + ' ' + name
            matchCat[newname] = np.nan

    for band in bandlabels:
        bandCat, _ = io.readFlareTbl(star, inst, band)
        bandRanges = np.array([bandCat['start'], bandCat['stop']]).T
        for i, flare in enumerate(masterCat):
            if flarecut is None and not flare['flare']:
                continue
            if flarecut is not None and flare['PEWratio'] < flarecut:
                continue
            masterRange = np.array([[flare['start'], flare['stop']]])
            duration = flare['stop'] - flare['start']
            overlapping, _ = utils.argoverlap(bandRanges, masterRange, 'loose')
            slimCat = bandCat[overlapping & bandCat['flare']]
            overranges = zip(slimCat['start'], slimCat['stop'])
            overranges = mnp.range_intersect(overranges, [masterRange])
            overlap = overranges[:,1] - overranges[:,0]
            if np.any(overlap/duration) > 0.0:
                j = np.argmax(overlap)
                for name in bandCat.colnames:
                    matchCat[i][band + ' ' + name] = slimCat[j][name]

    return matchCat


def combine_flarecats(bandname, inst, flarecut=1.0, stars='all'):
    fs = db.findfiles(rc.flaredir, inst, bandname, 'flares', fullpaths=True)
    if stars != 'all':
        hasStar = lambda f: any([s in f for s in stars])
        fs = filter(hasStar, fs)

    tbls, dts, bands = [], [], []
    expt = 0.0
    for f in fs:
        star = db.parse_info(f, 3, 4)
        flareTable = Table.read(f, format='fits')
        expt += flareTable.meta['EXPTIME']

        # store start, stop, peak relative times under new names
        for key in ['start', 'peak', 'stop']:
            flareTable[key + ' rel'] = flareTable[key]

        # change begs and ends to mjd
        mjd0 = flareTable.meta['MJD0']
        flareTable['start'] = flareTable['start']/24.0/3600.0 + mjd0
        flareTable['stop'] = flareTable['stop']/24.0/3600.0 + mjd0
        flareTable['peak'] = flareTable['peak']/24.0/3600.0 + mjd0
        flareTable['start'].unit, flareTable['stop'].unit, flareTable['peak'].unit = 'mjd', 'mjd', 'mjd'

        # cull non-flares
        keep = flareTable['flare'] & (flareTable['PEWratio'] > flarecut)
        flareTable = flareTable[keep]

        # add some columns specific to table
        n = len(flareTable)
        bandname = db.parse_info(f, 4, 5)
        flareTable['inst'] = Table.Column([inst]*n, description='instrument', unit='', dtype=str)
        flareTable['star'] = Table.Column([star]*n, unit='', dtype=str)
        flareTable['bandname'] = Table.Column([bandname]*n, unit='', dtype=str)

        # record dt and bands to check that all are the same later
        dts.append(flareTable.meta['DT'])
        keys = flareTable.meta.keys()
        nbands = sum(map(lambda s: 'BANDBEG' in s, keys))
        band0 = [flareTable.meta['BANDBEG' + str(i)] for i in range(nbands)]
        band1 = [flareTable.meta['BANDEND' + str(i)] for i in range(nbands)]
        bands.append(np.array(zip(band0, band1)))

        # metadata will just cause merge conflicts
        for key in flareTable.meta:
            del flareTable.meta[key]
        del flareTable['flare']

        tbls.append(flareTable)

    assert np.allclose(dts, dts[0])
    assert np.allclose(bands[:-1], bands[1:])

    # compute cum. freq. of flare PEWs for all stars
    tbl = vstack(tbls)
    tbl.sort('PEW')
    tbl.reverse()
    tbl.meta['EXPTIME'] = expt
    tbl.meta['DT'] = dts[0]
    tbl.meta['BANDS'] = bands[0]
    tbl['cumfreqPEW'] = cumfreq(tbl['PEW'], expt)
    tbl['cumfreqPEW'].description = 'cumuluative frequency of all flares with >= PEW for the instrument'
    tbl['cumfreqE'] = cumfreq(tbl['energy'], expt)
    tbl['cumfreqE'].description = 'cumuluative frequency of all flares with >= energy for the instrument'
    tbl['cumfreqPEW'].unit = tbl['cumfreqE'].unit = 'd-1'

    return tbl


def auto_curve(star, inst, bands, dt, appx=True, groups=None, fluxed=False):
    ph, photons = io.readphotons(star, inst)

    # make bins
    gtis = ph['gti'].data
    tbinList = []
    for t0, t1 in zip(gtis['start'], gtis['stop']):
        if appx:
            n = round((t1 - t0) / dt)
            if n == 0: n = 1
            tbinList.append(np.linspace(t0, t1, n+1))
        else:
            tbinList.append(np.arange(t0, t1, dt))
    badbins = np.cumsum([len(b) for b in tbinList])[:-1] - 1
    tbins = np.hstack(tbinList)

    eps = 'epera' if fluxed else 'epsilon'
    p = photons
    tedges, rate, err = sp.spectral_curves(p['time'], p['wavelength'], eps=p[eps], tbins=tbins, bands=bands,
                                   groups=groups)
    t0, t1 = [], []
    for i in range(len(tedges)):
        rate[i], err[i] = [np.delete(a, badbins) for a in [rate[i], err[i]]]
        t0.append(np.delete(tedges[i][:-1], badbins))
        t1.append(np.delete(tedges[i][1:], badbins))

    assert all([np.all((tt1-tt0) < dt*2.0) for tt0,tt1 in zip(t0, t1)])
    return t0, t1, rate, err


def __inrange(spectbl, wr):
    in0, in1 = [mnp.inranges(spectbl[s], wr) for s in ['w0', 'w1']]
    return spectbl[in0 | in1]


def __edgesinrange(spectbl, wr):
    w = mnp.lace(spectbl['w0'], spectbl['w1'])
    duplicates = np.append((w[:-1] == w[1:]), False)
    w = w[~duplicates]
    return w[mnp.inranges(w, wr)]


def __same_instrument(spectbls):
    instruments = []
    for s in spectbls: instruments.extend(s['instrument'].data)
    instruments = np.array(instruments)
    if any(instruments != instruments[0]):
        raise ValueError('There are multiple instruments present in the '
                         'spectbls.')
    return instruments[0]


def __make_masks(welist, dqlist, dqmasks):
    # make master grid
    mwe = specutils.common_grid(welist)

    # rebin dq flags onto master grid, make masks, coadd those
    mwe_ins = [mnp.inranges(mwe, we[[0, -1]]) for we in welist]
    mwelist = [mwe[mwe_in] for mwe_in in mwe_ins]
    rdqs = map(mnp.rebin, mwelist, welist, dqlist, ['or'] * len(welist))
    masks = [(rdq & dqmask) > 0 for rdq, dqmask in zip(rdqs, dqmasks)]

    mmask = np.ones(len(mwe) - 1, bool)
    for mask, mwe_in in zip(masks, mwe_ins):
        i = np.nonzero(mwe_in)[0][:-1]
        mmask[i] = mmask[i] & mask

    # find the ranges where every spectrum is masked
    wbins = np.array([mwe[:-1], mwe[1:]]).T
    badranges = specutils.flags2ranges(wbins, mmask)

    # set each mask to false over those ranges
    masks = [(dq & dqmask) > 0 for dq, dqmask in zip(dqlist, dqmasks)]
    for we, mask in zip(welist, masks):
        inbad0, inbad1 = [mnp.inranges(w, badranges) for w in [we[:-1], we[1:]]]
        inbad = inbad0 | inbad1
        mask[inbad] = False

    return masks


def __same_star(spectbls):
    stars = np.array([s.meta['STAR'] for s in spectbls])
    if any(stars[1:] != stars[:-1]):
        raise ValueError('More than one target in the provided spectra.')
    return stars[0]
