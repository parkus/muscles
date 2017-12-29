import netCDF4 as nc
from .. import rc
from os import path
import numpy as np
import mypy.my_numpy as mnp
from astropy.table import Table
from scipy.io import readsav
import astropy.units as u
from astropy.io import fits

def SEEFlareRatios(proximityCut=0.1):
    ## get the data from the file
    L3Afile = path.join(rc.solarpath, 'u_tmd_see_-----_sun_spectra_L3A.ncdf')
    with nc.Dataset(L3Afile) as ncsee:
        data = ncsee.variables

        # obs times
        year = np.floor_divide(data['DATE'][0], 1000)
        day = data['DATE'][0] - year*1000 # out of 365
        sec = data['TIME'][0]
        time = year + day/365.0 + sec/3600.0/24.0/365.0

        # spectra
        flux = data['SP_FLUX'][0] / 10.0 # W m-2 AA-1
        err = data['SP_ERR_MEAS'][0] * flux # msmt precision vs abs error

        # line data
        lineflux = data['LINE_FLUX'][0] / 10.0  # W m-2 AA-1
        lineerr = data['LINE_ERR_MEAS'][0] / 10.0 * lineflux
        linewaves = data['LINEWAVE'][0] * 10.0 # AA
        linenames = [''.join(n) for n in data['LINENAME'][0]]
        linenames = ['{} {:.0f}'.format(n, w) for n, w in zip(linenames, np.floor(linewaves))]

    ## compute continuum fluxes
    # defined indices just by looking at the data
    # I matched the cont bands from the variability paper as closely as possible (there were 1340-1350, 1372-1380.5,
    # 1382.5-1389, and 1413-1424.5
    contnames = ['continuum 1340-1420']
    contindices = [[134, 137, 138, 141]]
    contflux =  [np.sum(flux[:, i], axis=1) * 10.0 for i in contindices] # W m-2
    conterr = [mnp.quadsum(err[:, i], axis=1) * 10.0 for i in contindices] # W m-2
    contflux, conterr = [np.array(a).T for a in [contflux, conterr]]

    ## open and parse data from SEE flare catalog
    flareCat = Table.read(path.join(rc.solarpath, 'u_tmd_see_-----_sun_flare_catalog.csv'))

    # exclude events where SEE obs is too far from peak
    startsec = flareCat['start hour']*3600 + flareCat['start min']*60
    stopsec = flareCat['stop hour']*3600 + flareCat['stop min']*60
    # some spill over to next day
    stopsec[stopsec < startsec] = stopsec[stopsec < startsec] + 3600*24.0
    duration = stopsec - startsec
    keep = flareCat['see lag'].astype('f')/duration < proximityCut
    slimCat = flareCat[keep]

    ## combine line and cont data for flares
    names = linenames + contnames
    fluxes = np.hstack([lineflux, contflux])
    errs = np.hstack([lineerr, conterr])

    ## find which see observations are closest to the peak of each flare
    # get time of flare peak in decimal years
    peakyear = slimCat['year'] + slimCat['day of year']/365.0 + slimCat['peak hour']/365.0/24.0 \
                 + slimCat['peak min']/365.0/24.0/60.0
    startyear = slimCat['year'] + slimCat['day of year']/365.0 + slimCat['start hour']/365.0/24.0 \
                 + slimCat['start min']/365.0/24.0/60.0

    # for each peak, find the closest point and save the flux values for each line
    iflares = []
    for peak in peakyear:
        lags = time - peak
        iflares.append(np.argmin(np.abs(lags)))

    # then find nearest quiescent point with good S/N before the flare and take ratio
    goodSN = fluxes/errs > 5.0
    ratioList, ratioerrList = [], []
    for i in range(len(names)):
        ratios, ratioerrs = [], []
        for start, iflare in zip(startyear, iflares):
            b4flare = time < start
            usable = b4flare & goodSN[:, i]
            if np.any(usable):
                iquiescent = np.max(np.nonzero(usable)[0])
                ratio = fluxes[iflare, i]/fluxes[iquiescent, i]
                ratios.append(ratio)
                error = ratio * np.sqrt((errs[iquiescent, i]/fluxes[iquiescent, i])**2
                                        + (errs[iflare, i]/fluxes[iflare, i])**2)
                ratioerrs.append(error)
            else:
                ratios.append(np.nan)
                ratioerrs.append(np.nan)
        ratioList.append(ratios)
        ratioerrList.append(ratioerrs)
    ratioList = map(np.array, ratioList)
    ratioerrList = map(np.array, ratioerrList)
    return dict(zip(names, ratioList)), dict(zip(names, ratioerrList))


def EVEFlareCat2FITS(savdata=None):

    if savdata is None:
        print 'Reading in the data. Note that this takes a minute or two.'
        savdata = readsav(path.join(rc.solarpath, 'u_sdo_eve_-----_sun_flare_catalog.sav'))

    EVEcat = savdata['flare_catalog']['EVL']

    def parsetime(flare):
        idstr = flare['flare_id'][0]
        return float(idstr[:4])*365.0 + float(idstr[4:7])
    expt = parsetime(EVEcat[-1]) - parsetime(EVEcat[0]) #days

    # I'll make a separate FITS table for each of these
    linelabels = EVEcat[0]['evl_lines'][0]['evl_label']

    # Those tables will contain these columns
    keys = ['preflare_irrad', 'peak_irrad', 'peak_time_jd', 'rise_25_time_jd', 'rise_50_time_jd', 'rise_75_time_jd',
            'decay_25_time_jd', 'decay_50_time_jd', 'decay_75_time_jd', 'energy_25', 'energy_50', 'energy_75']
    # renamed to (norm_pk and cumfreq will be added later)
    colnames = ['flux_pre', 'flux_pk', 'tpeak', 'trise25', 'trise50', 'trise75',
                'tdecay25', 'tdecay50', 'tdecay75', 'energy25', 'energy50', 'energy75',
                'pew', 'norm_pk', 'Pcumfreq', 'Ecumfreq']
    # with the following units and data types
    units = ['erg cm-2']*2 + ['JD']*7 + ['erg']*3 + ['s', ''] + ['d-1']*2
    dtypes = ['D']*16

    # loop through and make each table
    fitstbls = []
    for iline in range(len(linelabels)):

        # make dictionary of empty lists for appending to
        cols = {}
        for key in keys:
            cols[key] = []

        # parse the data for the line for all flares
        for flare in EVEcat:
            flare = flare['evl_lines'][0]
            for key in keys:
                cols[key].append(flare[iline][key])

        # add some stuff...
        cols['peak_norm'] = np.array(cols['peak_irrad'])/cols['preflare_irrad']
        cols['pew'] = np.array(cols['energy_75'])/cols['preflare_irrad']

        # groom the columns a bit by converting to arrays and changing some units where desired
        for key in cols:
            cols[key] = np.array(cols[key])
            if 'energy' in key:
                cols[key] = cols[key] * u.J / u.m**2
                cols[key] = cols[key] * (4*np.pi*(u.AU)**2)
                cols[key] = cols[key].to(u.erg).value
            if 'irrad' in key:
                cols[key] = cols[key] * u.W / u.m**2
                cols[key] = cols[key].cgs.value

        # add cumulative frequency
        for ekey in ['energy_75', 'pew']:
            e = cols[ekey]
            bad = (e <= 0.0) | np.isnan(e)
            e[bad] = -np.inf
            argsort = np.argsort(cols[ekey])[::-1]  # bad flares will be sorted to end
            nflares = np.arange(len(EVEcat)) + 1.0
            cumfreq = nflares/expt
            cumfreq[argsort] = cumfreq.copy()
            cumfreq[bad] = np.nan
            cols[ekey[0].upper() + 'cumfreq'] = cumfreq

        # parse line info
        linelabel = linelabels[iline]
        pieces = linelabel.split(' ')
        name = ''.join(pieces[1:3])
        wave = float(pieces[3]) * 10.0 # AA
        Tform = float(pieces[-1][2:5])

        # now create a fits bintable
        hdr = fits.Header()
        hdr['ion'] = name
        hdr['wave'] = wave
        hdr['logTform'] = Tform
        arrays = [cols[key] for key in (keys + ['pew', 'peak_norm', 'Pcumfreq', 'Ecumfreq'])]
        colsFITS = [fits.Column(array=a, name=n, unit=ut, format=dt)
                    for a,n,ut,dt in zip(arrays, colnames, units, dtypes)]
        tbl = fits.BinTableHDU.from_columns(colsFITS, header=hdr)
        tbl.name = name
        fitstbls.append(tbl)

    # prepend a primary hdu and save the fits file
    pri = fits.PrimaryHDU()
    hdus = fits.HDUList([pri] + fitstbls)
    hdus.writeto(path.join(rc.solarpath, 'u_sdo_eve_-----_sun_flare_catalog.fits'), clobber=True)