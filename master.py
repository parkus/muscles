import muscles.database as db
import muscles.reduce as red

for star in db.stars:

    # rename files
#    db.auto_rename(db.datapath)
    
    # coadd spectra
    red.auto_coadd(star)
    
    # interpolate and save phoenix spectrum
    red.auto_phxspec(star)
    
    # make custom extractions
    red.auto_customspec(star)

    # make panspectrum
    specs = red.panspectrum(star, R=1000.0) #panspec and Rspec
