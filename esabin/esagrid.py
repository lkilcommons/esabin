# (C) 2019 University of Colorado AES-CCAR-SEDA (Space Environment Data Analysis) Group
# Liam Kilcommons - University of Colorado, Boulder - Colorado Center for Astrodynamics Research
import numpy as np
import h5py,os
from collections import OrderedDict
from esabin import spheretools

class EsagridBin(object):
    """Class which abstractly represents one bin in a grid"""
    def __init__(self,grid,flatind):
        self._meta = OrderedDict()
        self.grid = grid

        self['flatind'] = flatind
        self['slat'] = grid.flat_bins[flatind,0]
        self['elat'] = grid.flat_bins[flatind,1]
        self['lat'] = spheretools.angle_midpoint(self['slat'],
                                                    self['elat'],
                                                    grid.azi_units)
        self['sazi'] = grid.flat_bins[flatind,2]
        self['eazi'] = grid.flat_bins[flatind,3]
        self['azi'] = spheretools.angle_midpoint(self['sazi'],
                                                    self['eazi'],
                                                    grid.azi_units)

        #Alias azimuth coordinate by lon or lt
        self['s{}'.format(grid.azi_coord)]=self['sazi']
        self['e{}'.format(grid.azi_coord)]=self['eazi']
        self['{}'.format(grid.azi_coord)]=self['azi']

        self['azi_coord']=grid.azi_coord

        #Long name of bin
        self['longname'] = str(self)

    def __str__(self):
        return ('#%d,' % (self['flatind'])
                +'lat:%.3f-%.3f,' % (self['slat'],self['elat'])
                +'%s:%.3f-%.3f' % (self['azi_coord'],self['sazi'],self['eazi']))

    def __getitem__(self,key):
        return self._meta[key]

    def __setitem__(self,key,value):
        self._meta[key]=value

    def items(self):
        return self._meta.items()

    def __contains__(self,key):
        return key in self._meta

class Esagrid(object):
    """
    Equal solid angle binning class. Equal solid angle binning is a way of binning geophysical data
    (or other types of data which are naturally spherically located, lat, lon), which tackles the
    problem of bins of fixed latitude/longitude width covering different amounts of surface area.
    The bins produced by the Equal Solid Angle Scheme are guaranteed to each cover nearly the same
    surface area. Although many schemes which share this property are possible, for simplicity, this
    code implements only a fixed latitude width variety (that is, all bins have the same latitude span,
    but each band of latitude has a different number of longitudinal bins).
    """
    def __init__(self,delta_lat,n_cap_bins=3,azi_coord='lt'):
        """
        INPUTS
        ------
            delta_lat - float
                The latitudinal width of the bins to be produced

            n_cap_bins - integer, optional
                The number of bins touching the northern pole,
                (or southern since symmetric)/
                the minimum number of bins in a latitude band
                default is 3

            azi_coord - {'longitude','lon','localtime','lt','mlt'}, optional
                Which type of zonal/azimuthal/longitudinal coordinate to use

            rectangular - bool, optional
                Use a rectangular grid ()

        RETURNS
        -------
            Nothing
        """
        self.delta_lat = float(delta_lat)
        self.n_cap_bins = n_cap_bins
        self.azi_coord = azi_coord

        #Check if we have an integer number of bins
        if not np.equal(np.mod(180,self.delta_lat),0):
            errstr = ("Non-integer number of latitudinal(meridional) bands"
                      +"%.1f does not divide evenly into 180.\n" %(self.delta_lat))
            raise ValueError(errstr)

        self.n_lat_bins = int(180./self.delta_lat-1.)
        
        #Create the edges for each bin
        self.lat_bin_edges = np.linspace(-90.,90.,num=self.n_lat_bins+1)
        self.lon_bin_edges = []
        self.flat_bins = [] #A flat list of bins from south to north, -180 to 180 lon
                            #format is flat_bins[i] = [ith_bin_lat_start,ith_bin_lat_end,ith_bin_lon_start,ith_bin_lon_end]
        self.bin_map = []
        self.all_bin_inds = []

        #Determine conversion factor for radians to hours or degress
        #depending on whether we're using longitude or localtime
        if self.azi_coord in ['lon']:
            self.azi_fac = 180./np.pi
            self.max_azi = 180.
            self.min_azi = -180.
            self.azi_units = 'degrees'
        elif self.azi_coord in ['lt']:
            self.azi_fac = 12./np.pi
            self.max_azi = 12.
            self.min_azi = -12.
            self.azi_units = 'hours'
        else:
            raise ValueError(('Invalid azimuthal coordinate %s' % (self.azi_coord)
                              +'valid options ("lon" or "lt")'))
            
        flat_ind = 0 #the flat index of the nth lon bin in the mth lat band
        for m in range(len(self.lat_bin_edges)-1):
            #Compute the longitude bins for the mth latitude band
            self.lon_bin_edges.append(self.lonbins(self.lat_bin_edges[m],self.lat_bin_edges[m+1],self.n_cap_bins))

            #Add the the mth lon bin from the nth latitude band to the flat_bins and increment the flat_ind
            lat_band_flat_inds = [] #This lat band's flat index for each lon bin
            for n in range(len(self.lon_bin_edges[-1])-1):
                self.flat_bins.append([self.lat_bin_edges[m],self.lat_bin_edges[m+1],\
                                                    self.lon_bin_edges[-1][n],self.lon_bin_edges[-1][n+1]])
                #Add this bin index to this latitude band (for inclusion in
                #bin map)
                lat_band_flat_inds.append(flat_ind)

                #Add this bin index to the list of all bin incides
                self.all_bin_inds.append(flat_ind)

                flat_ind+=1

            #Add the flat indices of the longitude bins for the mth latitude band to the the bin_map
            #this must be an array so that which_bin can index it with the output
            #of np.digitize
            self.bin_map.append(np.array(lat_band_flat_inds))

        #Store total number of bins
        self.n_bins = flat_ind

        #Convert rectangular 4 x n_bins list of lat / azi limits of bins to arr
        self.flat_bins = np.array(self.flat_bins)

        #Convert list of all bin flat indices into an array
        self.all_bin_inds = np.array(self.all_bin_inds)

    def __str__(self):
        strrep = ''
        strrep += 'Equal Solid Angle Binning Grid\n'
        strrep += 'Latitude Spacing {} (deg)\n'.format(self.delta_lat)
        strrep += '{} Longitude Bins at Poles'.format(self.n_cap_bins)
        strrep += 'Azimuthal Coordinate {}\n'.format(self.azi_coord)
        strrep += 'Azimuth Range {} - {}\n'.format(self.min_azi,self.max_azi)
        strrep += '{} total bins'.format(np.nanmax(self.all_bin_inds))
        return strrep

    def azirangecheck(self,arr):
        #fmod preserves sign of input
        toobig = arr>(self.max_azi-self.min_azi)
        arr[toobig] = arr[toobig]-(self.max_azi-self.min_azi)
        arr[arr<self.min_azi] = arr[arr<self.min_azi] + (self.max_azi-self.min_azi)
        return arr

    def make_binable_lat_copy(self,lat):
        #check that lat is sane
        badlat = np.logical_or(lat>90.,
            lat<-90.)
        if np.count_nonzero(badlat)>0.:
            raise ValueError('Invalid latitude (|lat|>90.) values: %s' % (str(lat[badlat])))

        lat_to_bin = lat.copy()

        #Doesn't handle exactly 90 latitude properly (< not <= in digitize?) so we
        #must fudge
        lat_to_bin[lat==90.]=89.95
        lat_to_bin[lat==-90.]=-89.95

        return lat_to_bin

    def make_binable_lonorlt_copy(self,lonorlt):
        """
        Make copy of azimuthal coordinate array which
        has been changed to have the expected sign convention
        (e.g. -180 to 180 longitude as opposed to 0-360)
        """

        #Check for azimuth values that wrap above or below the
        #the specified azi bounds
        lonorlt_to_bin = lonorlt.copy()

        above_max = lonorlt_to_bin > self.max_azi
        below_min = lonorlt_to_bin < self.min_azi
        lonorlt_to_bin[above_max] -= (self.max_azi-self.min_azi)
        lonorlt_to_bin[below_min] += (self.max_azi-self.min_azi)

        #Check for azimuth values exactly at the maximum and minimum values
        equal_max = lonorlt_to_bin == self.max_azi
        equal_min = lonorlt_to_bin == self.min_azi
        lonorlt_to_bin[equal_max] -= (self.max_azi-self.min_azi)/1000.
        lonorlt_to_bin[equal_min] += (self.max_azi-self.min_azi)/1000.

        return lonorlt_to_bin

    def whichbin(self,lat,lonorlt):
        """Find which bin each (lat,lon/localtime) location is in
        
        PARAMETERS
        ----------
            lat - np.ndarray, shape=(n,)
                Array of 'n' latitudes
            lonorlt - np.ndarray, shape=(n,)
                Array of 'n' azimuthal coordinates (longitudes or localtimes)

        RETURNS
        -------
            latbands - np.ndarray, dtype=int, shape=(n,)
                Array of index of the latitude band of each location
            lonbins - np.ndarray, dtype=int, shape=(n,)
                Array of index of longitude bin within the latitude band
            flat_inds 
        """

        #This is nessecary to make sure that the input data
        #is not accidently changed inplace when the range of latitude
        #and azimuth are adjusted to match the assumptions made when generating
        #the bin limits
        lat_to_bin = self.make_binable_lat_copy(lat)
        lonorlt_to_bin = self.make_binable_lonorlt_copy(lonorlt)

        latbands = np.digitize(lat_to_bin,self.lat_bin_edges)-1
        #the -1 is because it returns 1 if bins[0]<x<bins[1]

        #Figure out which latbands have points so we don't have to search all of them
        unique_latbands = np.unique(latbands)

        flatinds = np.zeros(lat_to_bin.shape,dtype=int)
        lonbins = np.zeros_like(latbands)

        for band_ind in unique_latbands:
            in_band = latbands==band_ind
            lonbins[in_band] =  np.digitize(lonorlt_to_bin[in_band],self.lon_bin_edges[band_ind])-1
            flatinds[in_band] = self.bin_map[band_ind][lonbins[in_band]]

        return latbands,lonbins,flatinds

    def binarea(self,bindims,r_km=6371.2+110.):
        """
        Simply returns the bin surface area in km,
        assuming that the bin is a spherical
        rectangle on a sphere of radius r_km kilometers

        INPUTS
        ------
            bindims - [lat_start,lat_end,lonorlt_start,lonorlt_end]

            r_km - float, optional
                The radius of the sphere, in kilometers, defaults
                to Re+110km, i.e. the hieght of the ionosphere

        RETURNS
        -------
            A - float
                The surface area of the patch in km**2

        """
        theta2 = (90.-bindims[0])*np.pi/180. #theta / lat - converted to radians
        theta1 = (90.-bindims[1])*np.pi/180. #theta / lat - converted to radians
        dazi = spheretools.angle_difference(bindims[3],bindims[2])
        dphi = np.abs(dazi)/self.azi_fac #delta phi / lon - converted to radians
        return np.abs(r_km**2*dphi*(np.cos(theta1)-np.cos(theta2)))

    def lonbins(self,lat_start,lat_end,n_cap_bins):
        """
        Finds the longitudinal boundaries of the bins which
        have latitude boundaries lat_start and lat_end.
        The number of bins per latitude band is determined
        by scaling the number of bins touching the northern pole (n_cap_bins),
        i.e. the smallest possible number of bins in a latitude band.
        The scaling factor is derived by staring from the differential
        form of the solid angle:

        d(solid_angle) = sin(colat)d(colat)d(longitude)

        Then performing the integration around all longitudes
        and from colat_1 to colat_2, to get the total
        solid angle of a band from colat_1 to colat_2.

        Then a ratio is formed between the solid angle of an arbitrary
        band and the solid angle encompassed by the cap, i.e. the
        band between 0 colat and abs(colat_2-colat_1):

        2*pi/N_min*(1-cos(colat_2-colat_1))=2*pi/N_(1,2)*(cos(colat_2)-cos(colat_1))

        Finally, N_(1,2), the number of bins between colats 1 and 2 is solved for,
        as a function of N_min, the number of bins in the cap band.

        Of course, this formula is not guaranteed to produce an integer,
        so the result is rounded, producing close to, but not exactly,
        equal solid angle bins. Larger values of n_cap_bins produce
        more nearly exactly the same solid angle per bin.
        """
        th1 = (90.-lat_start)*np.pi/180.
        th2 = (90.-lat_end)*np.pi/180.
        N12 = (np.cos(th1)-np.cos(th2))/(1-np.cos(th2-th1))*n_cap_bins
        N12 = np.abs(np.round(N12))
        #+1 because we want N12 bins so we need N12+1 edges
        bins = np.linspace(-1*np.pi,np.pi,num=N12+1)*self.azi_fac

        #Check for any out of range values
        #print bins

        return bins

    def bin_locations(self,center_or_edges='edges'):
        """
        Output the location of each of the n_bins=len(self.flat_bins) bins

        if center_or_edges == 'edges', returns n_bins x 4 array:
        [left_lat_edges,right_lat_edge,lower_longitude_lt_edge,upper_longitude_lt_edge]

        if center_or_edges == 'center', returns n_bins x 2 array:
        [center latitude of bin, center longitude/local time of bin]
        """
        if center_or_edges == 'edges':
            lat_edges = self.flat_bins[:,[0,1]]
            lonorlt_edges = self.azirangecheck(self.flat_bins[:,[2,3]])
            return lat_edges,lonorlt_edges
        elif center_or_edges == 'center':
            lat_centers = spheretools.angle_midpoint(self.flat_bins[:,0],
                                                        self.flat_bins[:,1],
                                                        'degrees')
            lonorlt_centers = spheretools.angle_midpoint(self.flat_bins[:,2],
                                                         self.flat_bins[:,3],
                                                         self.azi_units)
            lonorlt_centers = self.azirangecheck(lonorlt_centers)
            return lat_centers,lonorlt_centers
        else:
            raise ValueError('Invalid center_or_edges value %s' %(center_or_edges))

    def bin_stats(self,lat,lonorlt,data,statfun=np.mean,center_or_edges='edges'):

        latbands,lonbins,flat_inds = self.whichbin(lat,lonorlt)

        binstats = np.zeros((len(self.flat_bins[:,0]),))
        binstats[:] = np.nan

        populated_bins = np.unique(flat_inds)

        for bin_ind in populated_bins:
            in_bin = flat_inds == bin_ind
            binstats[bin_ind] = statfun(data[in_bin].flatten())

        binlats,binlonorlts = self.bin_locations(center_or_edges=center_or_edges)

        return binlats,binlonorlts,binstats
