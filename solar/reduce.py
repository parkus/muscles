import netCDF4 as nc
from .. import database as db
from os import path
import numpy as np
import mypy.my_numpy as mnp
from astropy.table import Table

def SEELineFluxes(proximityCut=0.1):
    ## get the data from the file
    L3Afile = path.join(db.solarpath, 'u_timed_see_L3A.ncdf')
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
        linenames = ['{} {:.0f}'.format(n, w) for n, w in zip(linenames, np.round(linewaves))]

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
    flareCat = Table.read(path.join(db.solarpath, 'u_see_flare_catalog.csv'))

    # exclude events where SEE obs is too far from peak
    startsec = flareCat['start hour']*3600 + flareCat['start min']*60
    stopsec = flareCat['stop hour']*3600 + flareCat['stop min']*60
    # some spill over to next day
    stopsec[stopsec < startsec] = stopsec[stopsec < startsec] + 3600*24.0
    duration = stopsec - startsec
    keep = flareCat['see lag'].astype('f')/duration < proximityCut
    slimCat = flareCat[keep]

    ## find which see observations are closest to the peak of each flare
    iflare = []
    peakyear = slimCat['year'] + slimCat['day of year']/365.0 + slimCat['peak hour']/365.0/24.0 \
                 + slimCat['peak min']/365.0/24.0/60.0
    for year in peakyear:
        lags = time - year
        iflare.append(np.argmin(np.abs(lags)))

    ## find which see observations are completely outside of any flare
    startyear = flareCat['year'] + flareCat['day of year']/365.0 + flareCat['start hour']/365.0/24.0 \
                 + flareCat['start min']/365.0/24.0/60.0
    stopyear = flareCat['year'] + flareCat['day of year']/365.0 + flareCat['stop hour']/365.0/24.0 \
                 + flareCat['stop min']/365.0/24.0/60.0
    # some spill over to next day
    stopyear[stopyear < startyear] = stopyear[stopyear < startyear] + 1.0/365.0
    flareranges = np.array(zip(startyear, stopyear))
    inflare = mnp.inranges(time, flareranges)
    quiescent = ~inflare

    ## combine line and cont data for flares
    names = linenames + contnames
    fluxes = np.hstack([lineflux, contflux])
    errs = np.hstack([lineerr, conterr])

    ## find mean fluxes from quiescent periods
    means, meanerrs = [], []
    for flux, err  in zip(fluxes.T, errs.T):
        good = flux/err > 10.0
        use = good & quiescent
        means.append(np.mean(fluxes[use]))
        meanerrs.append(np.std(fluxes[use]) / np.sqrt(np.sum(use)))
    means, meanerrs = map(np.array, [means, meanerrs])

    # compute flare/quiescent ratio for each line and cont region
    means = np.vstack([means]*len(iflare))
    meanerrs = np.vstack([meanerrs]*len(iflare))
    ratios = fluxes[iflare] / means
    ratioerrs = ratios * np.sqrt((errs[iflare]/fluxes[iflare])**2 + (meanerrs/means)**2)
    return dict(zip(names, ratios.T)), dict(zip(names, ratioerrs.T))