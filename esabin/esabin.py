# Liam Kilcommons - University of Colorado, Boulder - Colorado Center for Astrodynamics Research
# Originally created May, 2016
# (C) 2019 University of Colorado AES-CCAR-SEDA (Space Environment Data Analysis) Group
import numpy as np
import h5py,os
from collections import OrderedDict
from . import spheretools

class EsaGridFileDuplicateTimeError(Exception):
	pass

class InvalidFlatIndexError(Exception):
	pass

class InvalidBinGroupNameError(Exception):
	pass

class EsagridBinComparisonError(Exception):
	pass

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

	def __eq__(self,other):
		bin_edge_keys = ['slat','elat','sazi','eazi']
		edges_match = []
		for key in bin_edge_keys:
			edges_match.append(np.isclose(self[key],other[key],
											rtol=0.,atol=1.0e-8))
		return all(edges_match)

	def __getitem__(self,key):
		return self._meta[key]

	def __setitem__(self,key,value):
		self._meta[key]=value

	def items(self):
		return self._meta.items()

	def __contains__(self,key):
		return key in self._meta

class EsagridFileBingroup(object):
	"""Class which abstractly represents the H5 group which describes
	a bin's position and data"""
	def __init__(self,grid,flatind):
		self.esagrid_bin = EsagridBin(grid,flatind)
		self.groupname = self._group_name_to_flatind(flatind)

	def __getitem__(self,key):
		return self.esagrid_bin[key]

	def __setitem__(self,key,value):
		self.esagrid_bin[key]=value

	def __contains__(self,key):
		return key in self.esagrid_bin

	def items(self):
		return self.esagrid_bin.items()

	@staticmethod
	def _flatind_from_group_name(h5groupname):
		flatindstr = h5groupname.split('bin')[-1]
		try:
			flatind = int(flatindstr)
		except ValueError:
			raise InvalidBinGroupNameError(('{} '.format(h5groupname)
											+'is not a valid bin group name'))
		return flatind

	@classmethod
	def from_groupname(cls,grid,groupname):
		flatind = cls._flatind_from_group_name(groupname)
		return cls(grid,flatind)

	def _check_flatind(self,flatind):
		grid = self.esagrid_bin.grid
		if flatind not in grid.all_bin_inds:
			raise InvalidFlatIndexError(('No bin with flat index {}'.format(flatind)
							   			+'in grid {}'.format(str(grid))))

	def _group_name_to_flatind(self,flatind):
		self._check_flatind(flatind)
		return 'bin%d' % (flatind)

	def _check_bin_group_name(self,h5group):
		#H5 groups' name is their full path
		h5groupname = h5group.name.split('/')[-1]
		if self.groupname != h5groupname:
			raise RuntimeError(('H5 group name {} did not match'.format(h5groupname)
							   +'EsagridFileBingroup {}'.format(self.groupname)))

	def check_bin_group_metadata(self,h5group,fix=False):
		#Check that this group matches this object
		self._check_bin_group_name(h5group)

		#Check for old/missing metadata (e.g. flatind not an integer)
		for attrname,attrval in self.esagrid_bin.items():
			if attrname in h5group.attrs:
				if h5group.attrs[attrname]!=attrval:
					print('Group:{}'.format(h5group.name))
					print('Incorrect Attribute {}'.format(attrname))
					print('{}!={}'.format(attrval,
										h5group.attrs[attrname]))
					if fix:
						h5group.attrs[attrname]=attrval
						print('Fixed')
			else:
				print('Group:{}'.format(h5group.name))
				print('Missing Attribute {}'.format(attrname))
				if fix:
					h5group.attrs[attrname]=attrval
					print('Fixed')


	def get_h5group(self,h5f):
		if self.groupname not in h5f:
			h5grp = h5f.create_group(self.groupname)
			#Write bin describing metadata
			for attrname,attrval in self.esagrid_bin.items():
				h5grp.attrs[attrname]=attrval
		else:
			h5grp = h5f[self.groupname]
			self.check_bin_group_metadata(h5grp,fix=True)
		return h5grp

	def store(self,h5f,t,data,additional_attrs=None,silent=False):
		h5grp = self.get_h5group(h5f)

		if isinstance(t,np.ndarray):
			#If time is an array, use the first value in the bin
			#as the hdf5 dataset name
			h5datasetnm = str(t.flatten()[0])
		else:
			#If time is not an array, just use it's string
			#version as the dataset name
			h5datasetnm = str(t)

		#Ensure no dataset name collisions
		if h5datasetnm in h5grp:
			raise EsaGridFileDuplicateTimeError(('Dataset with name'
			                             	+' {}'.format(h5datasetnm)
			                             	+' already exists in'
			                             	+' group {}'.format(h5grp)))
		# else:
		# 	while h5datasetnm in h5grp:
		# 		h5datasetnm += '0'

		dataset = h5grp.create_dataset(h5datasetnm,data=data)
		if additional_attrs is not None:
			for attr in additional_attrs:
				dataset.attrs[attr]=additional_attrs[attr]

		if not silent:
			print("Added %d points to %s" % (np.count_nonzero(in_bin),
											h5grp.attrs['longname']))

	def copy(self,h5f,destination_esagrid_file_bingroup,destination_h5f):
		src_esagrid_bin = self.esagrid_bin
		dest_esagrid_bin = destination_esagrid_file_bingroup.esagrid_bin
		if src_esagrid_bin != dest_esagrid_bin:
			raise EsagridBinComparisonError(('Cannot copy because '
											+'destination bin metadata does '
											+'not match source bin metadata '
											+'{} != '.format(str(dest_esagrid_bin))
											+'{}'.format(str(src_esagrid_bin))))

		h5grp = self.get_h5group(h5f)
		other_h5grp = destination_esagrid_file_bingroup.get_h5group(destination_h5f)
		for dataset_name in h5grp:
			if dataset_name in other_h5grp:
				raise EsaGridFileDuplicateTimeError(('Error while copying. '
													+' Dataset with name '
					                             	+' {}'.format(h5datasetnm)
					                             	+' already exists in '
					                             	+' destination '
					                             	+' group {}'.format(h5grp)))
			#Only h5py Group objects have a copy method
			h5grp.copy(dataset_name,other_h5grp,name=dataset_name)

	def _columnify_additional_attrs(self,additional_attrs):
		"""Takes a list of dictionaries. Each dictionary in the
		list are the HDF5 dataset attributes from one dataset in this
		bin's group. Converts this list of dictionaries to an
		output dictionary of lists or arrays depending on the type of
		data encountered. The keys of the output dictionary include any
		keys encountered in an input dictionary. If a particular key is
		not in every input dictionary a fill value with be inserted
		The fill value is numpy.nan if the data is numeric,
		otherwise it is None"""
		keys = []
		typefuncs = []
		for attrdict in additional_attrs:
			for key in attrdict:
				if key not in keys:
					keys.append(key)
					try:
						dum = float(attrdict[key])
						typefuncs.append(float)
					except ValueError:
						typefuncs.append(str)

		outdict = {key:[] for key in keys}
		for attrdict in additional_attrs:
			for key,typefunc in zip(keys,typefuncs):
				if key in attrdict:
					outdict[key].append(typefunc(attrdict[key]))
				else:
					outdict[key].append(np.nan if typefunc is float else None)

		for key,typefunc in zip(keys,typefuncs):
			if typefunc is float:
				outdict[key]=np.array(outdict[key])
		return outdict

	def read(self,h5f):
		h5grp = self.get_h5group(h5f)
		times = []
		datasets = []
		additional_attrs = []
		for dset_timestr in h5grp:
			dataset_time = float(dset_timestr)
			data = h5grp[dset_timestr][:]
			times.append(dataset_time)
			datasets.append(data)
			additional_attrs.append({key:val for key,val in h5grp[dset_timestr].attrs.items()})
		additional_attrs = self._columnify_additional_attrs(additional_attrs)
		return times,datasets,additional_attrs

class EsagridFile(object):
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

		clobber - bool, optional
			If True, will delete and overwrite the HDF5 file specified as os.path.join(hdf5_local_dir,hdf5_filenm)
			if it exists.
	"""
	def __init__(self,hdf5_filenm,grid=None,hdf5_local_dir=None,clobber=False):

		if hdf5_local_dir is None:
			raise ValueError(('hdf5_local_dir kwarg is now mandatory '
				              +'and cannot be None'))

		self.hdf5dir = hdf5_local_dir
		self.hdf5filenm = hdf5_filenm
		self.h5fn = os.path.join(self.hdf5dir,self.hdf5filenm)

		#Default to grid of with 3 latitude bins if no grid passed
		default_grid = DefaultEsagrid()

		if os.path.exists(self.h5fn):
			if not clobber:
				self.grid = self.create_grid_from_metadata()
			else:
				os.remove(self.h5fn)
				self.grid = default_grid if grid is None else grid
				self.write_grid_metadata()
		else:
			self.grid = default_grid if grid is None else grid
			self.write_grid_metadata()

		self.binlats,self.binlonorlts = self.grid.bin_locations(center_or_edges='edges')

		self._bingroups = {}
		with h5py.File(self.h5fn) as h5f:
			for groupname in h5f:
				try:
					bingroup = EsagridFileBingroup.from_groupname(self.grid,
																	groupname)
				except InvalidBinGroupNameError:
					print(('Could not create esagrid_file_bingroup'
						   +'from h5 group {}'.format(groupname)))
					continue
				self._bingroups[bingroup['flatind']]=bingroup


	def __getitem__(self,flatind):
		return self._bingroups[flatind]

	def __setitem__(self,flatind,esagrid_file_bingroup):
		if not isinstance(esagrid_file_bingroup,EsagridFileBingroup):
			raise TypeError(('{}'.format(esagrid_file_bingroup)
							+' is not an EsagridFileBingroup'))
		self._bingroups[flatind] = esagrid_file_bingroup

	def __contains__(self,flatind):
		return flatind in self._bingroups

	def items(self):
		return self._bingroups.items()

	def __iter__(self):
		for flatind in self._bingroups:
			yield flatind

	def write_grid_metadata(self):
		with h5py.File(self.h5fn) as h5f:
			h5f.attrs['delta_lat'] = self.grid.delta_lat
			h5f.attrs['n_cap_bins'] = self.grid.n_cap_bins
			h5f.attrs['azi_coord'] = self.grid.azi_coord

	def create_grid_from_metadata(self):
		with h5py.File(self.h5fn) as h5f:
			delta_lat = h5f.attrs['delta_lat']
			n_cap_bins = h5f.attrs['n_cap_bins']
			try:
				azi_coord = str(h5f.attrs['azi_coord'],'utf8')
			except:
				azi_coord = h5f.attrs['azi_coord']

		return Esagrid(delta_lat,n_cap_bins=n_cap_bins,azi_coord=azi_coord)

	def bin_and_store(self,t,lat,lonorlt,data,silent=False,additional_attrs=None):

		latbands,lonbins,flatinds = self.grid.whichbin(lat,lonorlt)

		populated_bins = np.unique(flatinds)

		with h5py.File(self.h5fn) as h5f:
			for bin_ind in populated_bins:

				in_bin = flatinds == bin_ind

				if bin_ind not in self:
					self[bin_ind] = EsagridFileBingroup(self.grid,bin_ind)

				self[bin_ind].store(h5f,
									t[in_bin].flatten(),
									data[in_bin].flatten(),
									additional_attrs=additional_attrs,
									silent=silent)

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

					try:
						flatind = EsagridFileBingroup._flatind_from_group_name(groupnm)
					except InvalidBinGroupNameError:
						print('{} is not a bin group, skipping'.format(groupnm))
						continue

					esagrid_file_bingroup = EsagridFileBingroup(self.grid,flatind)
					#Load h5 group and check for old or missing metadata
					grp = esagrid_file_bingroup.get_h5group(h5f)

					#Skip bins below the desired latitude
					if np.abs(grp.attrs['slat'])<minlat and np.abs(grp.attrs['elat'])<minlat:
						#if not silent:
						#	print("Skipping bin %s because too low latitude (<%.3f)" % (grp.attrs['longname'],minlat))
						continue

					statusstr = "| %s | " % (grp.attrs['longname'])
					flatind = grp.attrs['flatind']
					if np.floor(flatind) != flatind or flatind < 0:
						raise ValueError('Unexpected bin index {}'.format(flatind))
					flatind = int(flatind)

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

			azi_coord - {'lon','lt'}, optional
				Which type of zonal/azimuthal/longitudinal coordinate to use
				('lon' for longitude, 'lt' forlocaltime)

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
		self.lat_bin_edges = np.linspace(-90.,90.,num=int(self.n_lat_bins+1))
		self.lon_bin_edges = []
		self.flat_bins = [] #A flat list of bins from south to north, -180 to 180 lon
							#format is flat_bins[i] = [ith_bin_lat_start,ith_bin_lat_end,ith_bin_lon_start,ith_bin_lon_end]
		self.bin_map = []
		self.all_bin_inds = []

		#Determine conversion factor for radians to hours or degress
		#depending on whether we're using longitude or localtime
		if self.azi_coord == 'lon':
			self.azi_fac = 180./np.pi
			self.max_azi = 180.
			self.min_azi = -180.
			self.azi_units = 'degrees'
		elif self.azi_coord == 'lt':
			self.azi_fac = 12./np.pi
			self.max_azi = 12.
			self.min_azi = -12.
			self.azi_units = 'hours'
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

	def __eq__(self,other):
		if not isinstance(other,esagrid):
			raise TypeError('{} is not an esagrid instance'.format(other))
		return (self.delta_lat==other.delta_lat\
				 and self.n_cap_bins==other.n_cap_bins\
				 and self.azi_coord==other.azi_coord)

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
		"""
		Returns the flat index and the 2 digit index (lat_band_number, lon_bin_of_lat_band_number)
		of each (lat,lonorlt) pair.
		"""

		lat_to_bin = self.make_binable_lat_copy(lat)
		lonorlt_to_bin = self.make_binable_lonorlt_copy(lonorlt)

		latbands = np.digitize(lat_to_bin,self.lat_bin_edges)-1
		#the -1 is because it returns 1 if bins[0]<x<bins[1]

		#Figure out which latbands have points so we don't have to search all of them
		unique_latbands = np.unique(latbands)

		flat_inds = np.zeros(lat_to_bin.shape,dtype=int)
		lonbins = np.zeros_like(latbands)

		for band_ind in unique_latbands:
			in_band = latbands==band_ind
			lonbins[in_band] =  np.digitize(lonorlt_to_bin[in_band],self.lon_bin_edges[band_ind])-1
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
		bins = np.linspace(-1*np.pi,np.pi,num=int(N12+1))*self.azi_fac

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

class DefaultEsagrid(Esagrid):
	"""Default settings of a 3 degrees latitude per band, 3 cap bins, with
	localtime as the azimuthal coordinate"""
	def __init__(self,delta_lat=3,n_cap_bins=3,azi_coord='lt'):
		Esagrid.__init__(self,delta_lat,n_cap_bins=n_cap_bins,azi_coord=azi_coord)
