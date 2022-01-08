import pytest
import tempfile
import shutil,tempfile,os
import numpy as np
import h5py
from numpy.random import default_rng
from esabin.esagrid import Esagrid,ConstantAzimuthalSpacingGrid,EsagridBin

class SyntheticDatasource(object):
    """Creates synthetic data to test binning"""
    def __init__(self,azicoord,timetype,range_check_azi : bool):
        if azicoord not in ['lon','lt']:
            raise ValueError(('Invalid azicoord {}'.format(azicoord)
                              +'valid values lon or lt'))
        if timetype not in ['scalar','vector']:
            raise ValueError(('Invalid timetype {}'.format(timetype)
                              +'valid values scalar or vector'))
        self.azicoord = azicoord
        self.timetype = timetype
        self.range_check_azi = range_check_azi

    def walk_trajectory(start_lat,start_lon,n_samples):
        lat = start_lat
        lon = start_lon
        dlat = -.1
        dlon = -.3
        for isamp in range(n_samples):
            lat+=dlat
            lon+=dlon
            if np.abs(lat)>90.:
                lat+=np.sign(lat)*-180.
            if np.abs(lon)>180. and self.range_check_azi:
                lon+=np.sign(lon)*-360.
            yield lat,lon

    def __call__(self,n_samples):
        if timetype == 'scalar':
            t = 1.
        elif timetype == 'vector':
            t = np.arange(n_samples)

        walk_trajectory_args = (80.,75.,n_samples)
        lats,lons = np.zeros((n_samples,)),np.zeros((n_samples,))
        for isamp,(lat,lon) in enumerate(walk_trajectory(*walk_trajectory_args)):
            lats[isamp]=lat
            lons[isamp]=lon

        f = n_samples/4.
        omega = f/(2.*np.pi)
        data = np.sin(omega*t)

        if self.azicoord == 'lon':
            azis = lons
        elif self.azicoord == 'lt':
            azis = lons/180.*12.

        return t,lats,azis,data

def test_esagrid_raises_valueerror_on_bad_deltalat():
    """Esagrid constructor should error if it can't get an integer number
    of latitude bands with the given latitude spacing"""
    with pytest.raises(ValueError):
        delta_lat=.7
        grid = Esagrid(delta_lat)

def test_constantazimuthalspacinggrid_raises_valueerror_on_bad_deltaazi():
    """Esagrid constructor should error if it can't get an integer number
    of azimuthal bands with the given azimuthal spacing"""
    with pytest.raises(ValueError):
        delta_azi = .33
        grid = ConstantAzimuthalSpacingGrid(3.,delta_azi)


