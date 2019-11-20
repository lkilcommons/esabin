# Liam Kilcommons - University of Colorado, Boulder - Colorado Center for Astrodynamics Research
# Originally created May, 2016
# (C) 2019 University of Colorado AES-CCAR-SEDA (Space Environment Data Analysis) Group
import numpy as np
import h5py,os

class esagrid_file(object):
	"""
	Class for storing data associated with each bin of an esagrid object on disk.
	Uses h5py interface to HDF5 library.

	HDF5 files are organized thus:
	-Each bin gets a HDF5 group.
	-Each time the bin_and_store function is called, all of the groups/bins are iterated
		through and any data which falls within the bin's lat,lt or lat,lon boundaries is stored
		as a new dataset.

	INPUTS
	------
		hdf5_filenm - str
			The filename of the hdf5 file to store to. If this is an existing file
			and clobber == False, will use the stored metadata in the file to
			create the appropriate esagrid and you can continue adding to the file
			or process results from it

		grid - an esagrid instance, optional
			The grid of bins to bin into. If it is None (default), a default grid
			with delta_lat = 3 and n_cap_bins = 3 and azi_coord = 'lt' is used
		hdf5_local_dir - str, optional
			A valid local path at which the hdf5 files will be created
			if this is None (default) the geospacepy configuration setting
			config['esabin']['hdf5_local_dir'] will be used.
		clobber - bool, optional
			If True, will delete and overwrite the HDF5 file specified as os.path.join(hdf5_local_dir,hdf5_filenm)
			if it exists.
	"""
	def __init__(self,hdf5_filenm,grid=None,hdf5_local_dir=None,clobber=False):

		if hdf5_local_dir is None:
			raise ValueError(('hdf5_local_dir kwarg is now mandatory'
				              +'and cannot be None'))

		self.hdf5dir = hdf5_local_dir
		self.grid = grid
		self.hdf5filenm = hdf5_filenm
		self.h5fn = os.path.join(self.hdf5dir,self.hdf5filenm)
		if os.path.exists(self.h5fn) and clobber:
			os.remove(self.h5fn)
			if self.grid is None:
				self.grid = esagrid(3.)
			self.write_grid_metadata()
		elif os.path.exists(self.h5fn):
			self.create_grid_from_metadata()
		else:
			if self.grid is None:
				self.grid = esagrid(3.)
			self.write_grid_metadata()

		self.binlats,self.binlonorlts = self.grid.bin_locations(center_or_edges='edges')

	def write_grid_metadata(self):
		with h5py.File(self.h5fn) as h5f:
			h5f.attrs['delta_lat'] = self.grid.delta_lat
			h5f.attrs['n_cap_bins'] = self.grid.n_cap_bins
			h5f.attrs['azi_coord'] = self.grid.azi_coord

	def create_grid_from_metadata(self):
		with h5py.File(self.h5fn) as h5f:
			delta_lat = h5f.attrs['delta_lat']
			n_cap_bins = h5f.attrs['n_cap_bins']
			azi_coord = h5f.attrs['azi_coord']
			self.grid = esagrid(delta_lat,n_cap_bins=n_cap_bins,azi_coord=azi_coord)

	def get_bin_group(self,h5f,flatind):
		groupnm = 'bin%d' % (flatind)
		azi_coord = self.grid.azi_coord
		if groupnm not in h5f:
			h5f.create_group(groupnm)
			#Bin index
			h5f[groupnm].attrs['flatind']=flatind
			#Bin boundaries
			slat,elat,sazi,eazi = self.binlats[flatind,0],self.binlats[flatind,1],self.binlonorlts[flatind,0],self.binlonorlts[flatind,1]
			h5f[groupnm].attrs['slat']=slat
			h5f[groupnm].attrs['elat']=elat
			h5f[groupnm].attrs['s%s'%(azi_coord)]=sazi
			h5f[groupnm].attrs['e%s'%(azi_coord)]=eazi
			h5f[groupnm].attrs['longname'] = '#%d,lat:%.3f-%.3f,%s:%.3f-%.3f' % (flatind,slat,elat,azi_coord,sazi,eazi)
		return h5f[groupnm]

	def bin_and_store(self,t,lat,lonorlt,data,silent=False,additional_attrs=None):
		#check bounds of lonorlt
		lonorlt[lonorlt>self.grid.max_azi] = lonorlt[lonorlt>self.grid.max_azi]-(self.grid.max_azi-self.grid.min_azi)
		lonorlt[lonorlt<self.grid.min_azi] = lonorlt[lonorlt<self.grid.min_azi]+(self.grid.max_azi-self.grid.min_azi)

		latbands,lonbins,flat_inds = self.grid.whichbin(lat,lonorlt)

		populated_bins = np.unique(flat_inds)

		with h5py.File(self.h5fn) as h5f:
			for bin_ind in populated_bins:
				h5grp = self.get_bin_group(h5f,bin_ind)
				in_bin = flat_inds == bin_ind
				if isinstance(t,np.ndarray):
					#If time is an array, use the first value in the bin
					#as the hdf5 dataset name
					h5datasetnm = str(t[in_bin].flatten()[0])
				else:
					#If time is not an array, just use it's string
					#version as the dataset name
					h5datasetnm = str(t)

				this_data = data[in_bin].flatten()
				#Ensure no dataset name collisions
				while h5datasetnm in h5grp:
					h5datasetnm += '0'

				dataset = h5grp.create_dataset(h5datasetnm,data=this_data)
				if additional_attrs is not None:
					for attr in additional_attrs:
						dataset.attrs[attr]=additional_attrs[attr]

				if not silent:
					print("Added %d points to %s" % (np.count_nonzero(in_bin),h5grp.attrs['longname']))

	def dataset_passes_attr_filters(self,dataset,attr_filters,default_result=True):
		"""
		filtering whether to include a specific dataset in a bin_stats
		sample for a particular bin

		Filters are specified as a nested dictionary
		with the following grammar:
			filters['attr_key']=test_function

		where:
			'attr_key' : the NAME of the HDF5 attribute of the dataset
			test_function : a python lambda function or other function
							to apply to the value of attribute.
							This function must return a single True or
							False value and appropriately handle the
							type of data that is in the attribute

		"""
		passed_filters=default_result
		if attr_filters is not None:
			for attr_key in attr_filters:
				attr_test_fun = attr_filters[attr_key]
				if attr_key in dataset.attrs:
					attr_value = dataset.attrs[attr_key]
					passed_filters = passed_filters and attr_test_fun(attr_value)
		return passed_filters

	def bin_stats(self,statfun=np.nanmean,statfunname=None,
						center_or_edges='edges',minlat=50.,
						silent=False,force_recompute=False,
						write_to_h5=True,attr_filters=None):
		"""
			if statfun is list of functions, they will be applied succesively and binstats will be a list
			this is more time efficient than calling bin_stats multiple times
		"""
		if not isinstance(statfun,list):
			statfun = [statfun]

		if statfunname is not None:
			if not isinstance(statfunname,list):
				statfunname = [statfunname]

		#hdf5 dataset names where we can store results, or reload them if they've already been computed
		statfun_dataset_names = ['binstats_'+func.__name__ for func in statfun]
		print(statfun_dataset_names)

		binstats = []
		for fun in statfun:
			this_binstats = np.zeros((len(self.grid.flat_bins[:,0]),1))
			this_binstats.fill(np.nan)
			binstats.append(this_binstats)

		#Locate the bins
		binlats,binlonorlts = self.grid.bin_locations(center_or_edges=center_or_edges)

		with h5py.File(self.h5fn) as h5f:
			stats_computed = 'binstats_results' in h5f and \
				all([dataset_name in h5f['binstats_results'] \
					for dataset_name in statfun_dataset_names])
			if not force_recompute and stats_computed:
				if not silent:
					print("Loading precomputed results, set kwarg force_recompute=True to avoid this behaviour")
				for istatfun,statfun_dataset_name in enumerate(statfun_dataset_names):
					this_binstats = h5f['binstats_results'][statfun_dataset_name][:]
					if not silent:
						print("Loading %d nonzero bins from dataset %s..." % (np.count_nonzero(np.isfinite(this_binstats)),statfun_dataset_name))
					binstats[istatfun] = this_binstats

					#We can just short-circuit and return, since we don't need to do any more data loading
					#if len(binstats) == 1:
					#	binstats = binstats[0]
					#return binlats,binlonorlts,binstats
			elif not stats_computed or force_recompute:
				#Read every dataset (pass) from each bin
				for groupnm in h5f:
					grp = h5f[groupnm]
					#Skip the group if it doesn't have appropriate metadata:
					if 'slat' not in grp.attrs:
						print("Group %s does not have appropriate metadata" % (str(grp)))
						continue

					#Skip bins below the desired latitude
					if np.abs(grp.attrs['slat'])<minlat and np.abs(grp.attrs['elat'])<minlat:
						#if not silent:
						#	print("Skipping bin %s because too low latitude (<%.3f)" % (grp.attrs['longname'],minlat))
						continue

					statusstr = "| %s | " % (grp.attrs['longname'])
					flatind = grp.attrs['flatind']
					datasets = []
					ndatasets = len(grp.items())
					for datasetnm,dataset in grp.items():
						#Check that the dataset does not have
						#any attributes which are in the prohibited_attrs
						#list (an optional kwarg)
						is_okay = self.dataset_passes_attr_filters(dataset,
																attr_filters,
																default_result=True)
						if is_okay:
							datasets.append(dataset[:])

					statusstr+= "kept %d/%d passes | " % (len(datasets),ndatasets)
					if len(datasets)>0:
						bin_data = np.concatenate(datasets)
						#Do all stat functions
						for ifun,this_statfun in enumerate(statfun):
							binstats[ifun][flatind] = this_statfun(bin_data)
							statusstr+="%s: %.3f | " % (this_statfun.__name__,
														binstats[ifun][flatind])
					else:
						for ifun,this_statfun in enumerate(statfun):
							binstats[ifun][flatind] = np.nan
							statusstr+="%s: NaN | " % (this_statfun.__name__)

					if not silent:
						print(statusstr)
				#Save the results
			else:
				raise RuntimeError("Unhandled case!")

			if write_to_h5:
				for istatfun,statfun_dataset_name in enumerate(statfun_dataset_names):
					if 'binstats_results' not in h5f:
						h5f.create_group('binstats_results')
					results_grp = h5f['binstats_results']
					#Create a dataset for each statistics function's binned results
					if statfun_dataset_name in results_grp:
						del results_grp[statfun_dataset_name]
					results_grp.create_dataset(statfun_dataset_name,data=binstats[istatfun])
					if not silent:
						print("Saved binning results to h5 datatset %s" % (statfun_dataset_name))

		if statfunname is not None:
			#Return binstats as a dictionary if we named the statistics
			#functions
			binstats_dict = {}
			for i in range(len(binstats)):
				binstats_dict[statfunname[i]] = binstats[i]
			return binlats,binlonorlts,binstats_dict
		else:
			#Don't bother returning a list of results if we are only using one stat function
			#Just return the array
			if len(binstats) == 1:
				binstats = binstats[0]

			return binlats,binlonorlts,binstats

class esagrid(object):
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
		self.delta_lat = delta_lat
		self.n_lat_bins = int(180./self.delta_lat-1.)
		self.n_cap_bins = n_cap_bins
		self.azi_coord = azi_coord

		#Check if we have an integer number of bins
		if not np.equal(np.mod(self.n_lat_bins,1),0):
			print("Warning: Non-integer number of latitudinal(meridional) bands, %.1f does not divide evenly into 180.\n" %(self.delta_lat))
			print("A value of %.2f deg per bin will be used." % (180./np.floor(self.n_lat_bins+1.)))
			self.n_lat_bins = np.floor(self.n_lat_bins)

		#Create the edges for each bin
		self.lat_bin_edges = np.linspace(-90.,90.,num=self.n_lat_bins+1.)
		self.lon_bin_edges = []
		self.flat_bins = [] #A flat list of bins from south to north, -180 to 180 lon
							#format is flat_bins[i] = [ith_bin_lat_start,ith_bin_lat_end,ith_bin_lon_start,ith_bin_lon_end]
		self.bin_map = []

		#Determine conversion factor for radians to hours or degress
		#depending on whether we're using longitude or localtime
		if self.azi_coord in ['lon','Lon','LON','Longitude']:
			self.azi_fac = 180./np.pi
			self.max_azi = 180.
			self.min_azi = -180.
			self.azi_offset = 0.
		elif self.azi_coord in ['lt','mlt','LT','MLT','Localtime','localtime']:
			self.azi_fac = 12./np.pi
			self.max_azi = 12.
			self.min_azi = -12.
			self.azi_offset = 6.
		else:
			raise ValueError('Invalid azimuthal coordinate %s' % (self.azi_coord))

		flat_ind = 0 #the flat index of the nth lon bin in the mth lat band
		for m in range(len(self.lat_bin_edges)-1):
			#Compute the longitude bins for the mth latitude band
			self.lon_bin_edges.append(self.lonbins(self.lat_bin_edges[m],self.lat_bin_edges[m+1],self.n_cap_bins))

			#Add the the mth lon bin from the nth latitude band to the flat_bins and increment the flat_ind
			lat_band_flat_inds = [] #This lat band's flat index for each lon bin
			for n in range(len(self.lon_bin_edges[-1])-1):
				self.flat_bins.append([self.lat_bin_edges[m],self.lat_bin_edges[m+1],\
													self.lon_bin_edges[-1][n],self.lon_bin_edges[-1][n+1]])
				lat_band_flat_inds.append(flat_ind)
				flat_ind+=1

			#Add the flat indices of the longitude bins for the mth latitude band to the the bin_map
			self.bin_map.append(np.array(lat_band_flat_inds))


		#Convert rectangular lists of lists to arrays
		self.flat_bins = np.array(self.flat_bins)
		self.bin_map = np.array(self.bin_map)

	def azirangecheck(self,arr):
		#fmod preserves sign of input
		toobig = arr>(self.max_azi-self.min_azi)
		arr[toobig] = arr[toobig]-(self.max_azi-self.min_azi)
		arr[arr<self.min_azi] = arr[arr<self.min_azi] + (self.max_azi-self.min_azi)
		return arr

	def whichbin(self,lat,lonorlt):
		"""
		Returns the flat index and the 2 digit index (lat_band_number, lon_bin_of_lat_band_number)
		of each (lat,lonorlt) pair.
		"""
		#check that lat is sane
		badlat = np.logical_or(lat>90.,
			lat<-90.)
		if np.count_nonzero(badlat)>0.:
			raise ValueError('Invalid latitude (|lat|>90.) values: %s' % (str(lat[badlat])))

		#Doesn't handle exactly 90 latitude properly (< not <= in digitize?) so we
		#must fudge
		lat[lat==90.]=89.95
		lat[lat==-90.]=-89.95

		#check bounds of lonorlt
		lonorlt[lonorlt>self.max_azi] = lonorlt[lonorlt>self.max_azi]-(self.max_azi-self.min_azi)
		lonorlt[lonorlt<self.min_azi] = lonorlt[lonorlt<self.min_azi]+(self.max_azi-self.min_azi)

		latbands = np.digitize(lat,self.lat_bin_edges)-1 #the -1 is because it returns 1 if bins[0]<x<bins[1]
		#Figure out which latbands have points so we don't have to search all of them
		unique_latbands = np.unique(latbands)

		flat_inds = np.zeros_like(lat)
		lonbins = np.zeros_like(latbands)

		for band_ind in unique_latbands:
			in_band = latbands==band_ind
			lonbins[in_band] =  np.digitize(lonorlt[in_band],self.lon_bin_edges[band_ind])-1
			flat_inds[in_band] = self.bin_map[band_ind][lonbins[in_band]]

		return latbands,lonbins,flat_inds

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
		dphi = np.abs(bindims[3]-bindims[2])/self.azi_fac #delta phi / lon - converted to radians
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
		Get the location of each of the n_bins=len(self.flat_bins) bins

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
			lat_centers = np.mean(self.flat_bins[:,[0,1]],1)
			lonorlt_centers = self.azirangecheck(np.mean(self.flat_bins[:,[2,3]],1))
			return lat_centers,lonorlt_centers
		else:
			raise ValueError('Invalid center_or_edges value %s' %(center_or_edges))

	def eof_analysis(self,t,lat,lonorlt,data,n_eofs=2,center_or_edges='edges',method='cov'):
		"""
		Gets the first n_eofs emperical orthogonal functions
		for the data in data

		It's best to pass the orbit index in to t

		method - {'cov','corr'}
			Use covariance or correlation coefficient to compute EOFs
		"""

		#Bin the data (find out which bin each datapoint matches)
		latbands,lonbins,flat_inds = self.whichbin(lat,lonorlt)

		#Figure out which bins have data
		populated_bins = np.unique(flat_inds)

		#Make a list of boolean arrays which represent which data is in each bin
		data_in_each_bin = [flat_inds==bin_ind for bin_ind in populated_bins]

		#First divide the data into samples (make sure time is an integer using floor)
		data_in_each_sample = [np.floor(t)==this_t for this_t in np.unique(np.floor(t))]

		n,K = len(data_in_each_sample),len(data_in_each_bin)
		print("%d of %d bins have data. %d individual samples are detected. Data matrix will be %d x %d" % (len(data_in_each_bin),
			len(self.flat_bins),len(data_in_each_sample),n,K))

		#Initialize arrays
		data_array = np.zeros((len(data_in_each_sample),len(data_in_each_bin)))
		#Convert data_array to a masked array (so we can used masked covariance estimate function)
		data_array = np.ma.masked_array(data_array)
		populated_binmeans = np.zeros((len(data_in_each_bin)))
		binmeans = np.zeros((len(self.flat_bins),))
		bineofs = np.zeros((len(self.flat_bins),n_eofs))
		#Fill with NaNs
		data_array.fill(np.nan)
		binmeans.fill(np.nan)
		populated_binmeans.fill(np.nan)
		bineofs.fill(np.nan)

		#Make the data matrix with each of the K columns (variables)
		#representing a populated bin

		for ibin,data_in_bin in enumerate(data_in_each_bin):
			#Get the mean of each bin across all samples
			populated_binmeans[ibin] = np.nanmean(data[data_in_bin])
			for isample,data_in_sample in enumerate(data_in_each_sample):
				#Get the mean deviation from all-sample mean (the mean anomaly), for each sample and bin
				#This is what we will need to compute the covariance of, and the PCs will be the eigenvalues of
				#that covariance matrix
				data_in_bin_and_sample = np.logical_and(data_in_bin,data_in_sample)
				data_array[isample,ibin] = np.nanmean(data[data_in_bin_and_sample]-populated_binmeans[ibin])

		#Define the mask
		data_array.mask = np.logical_not(np.isfinite(data_array))

		#Compute the covariance matrix (if rowvar is not manually set np assumes that data is in columns,
		#but to keep with convention of Wilks book, we want data in rows and variables in columns)
		#print data_array


		if method == 'cov':
			S = np.ma.cov(data_array,rowvar=False)
			print("Computing %d x %d covariance matrix on %d samples"  % (K,K,n))
		elif method == 'corr':
			S = np.ma.corrcoef(data_array,rowvar=False)
			print("Computing %d x %d correlation coefficient matrix on %d samples"  % (K,K,n))
		else:
			raise ValueError('Bad EOF determination method, choose "cov" for covariance, "corr" for correlation coefficient')

		#The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i].
		#Since covariance matrices are by definition symmetric we can get eigs with a
		#specific function (for general matices use np.linalg.eig)
		#Eigen vectors come out in order of size, and with degenerate eigenvectors repeated
		w,v = np.linalg.eigh(S)

		#Compute the percentage of variance represented by each eigenvector
		Rsquared = w/np.nansum(w)*100.
		print("Percent variance represented by each of 2*n_eofs eigenvectors:\n%s" % (str(Rsquared[:int(n_eofs*2)])))

		#Assemble the n_eofs eofs with largest variance
		#(bineofs has length == total number of bins)
		bineofs[populated_bins.astype(int),:] = v[:,-n_eofs:]

		#Similarly for the bin means (binmeans has length == total number of bins)
		binmeans[populated_bins.astype(int)] = populated_binmeans

		#Find the location of the bins
		binlats,binlonorlts = self.bin_locations(center_or_edges=center_or_edges)

		return binlats,binlonorlts,binmeans,bineofs,Rsquared[-n_eofs:]

	def bin_stats(self,lat,lonorlt,data,statfun=np.mean,center_or_edges='edges'):

		#check bounds of lonorlt
		lonorlt[lonorlt>self.max_azi] = lonorlt[lonorlt>self.max_azi]-(self.max_azi-self.min_azi)
		lonorlt[lonorlt<self.min_azi] = lonorlt[lonorlt<self.min_azi]+(self.max_azi-self.min_azi)

		latbands,lonbins,flat_inds = self.whichbin(lat,lonorlt)

		binstats = np.zeros((len(self.flat_bins[:,0]),1))
		binstats[:] = np.nan

		populated_bins = np.unique(flat_inds)

		for bin_ind in populated_bins:
			in_bin = flat_inds == bin_ind
			binstats[bin_ind] = statfun(data[in_bin].flatten())

		binlats,binlonorlts = self.bin_locations(center_or_edges=center_or_edges)

		return binlats,binlonorlts,binstats
