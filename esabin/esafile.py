# (C) 2019 University of Colorado AES-CCAR-SEDA (Space Environment Data Analysis) Group
# Liam Kilcommons - University of Colorado, Boulder - Colorado Center for Astrodynamics Research
# Originally created May, 2016
import numpy as np
import h5py,os,shutil
from collections import OrderedDict
from esabin.esagrid import Esagrid,ConstantAzimuthalSpacingGrid,EsagridBin
from esabin import spheretools

class EsaGridFileDuplicateTimeError(Exception):
	pass

class InvalidFlatIndexError(Exception):
	pass

class InvalidBinGroupNameError(Exception):
	pass

class EsagridBinComparisonError(Exception):
	pass

class EsagridFileBinGroup(object):
	"""Class representing an HDF5 Group containing data from one bin
	
	PARAMETERS
	----------
	grid : esabin.esagrid.ConstantLatitudeSpacingGrid
		Object representing the particular grid used
	flatind : int
		The index (number) defining which bin this Group is for
	"""
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
		if flatind < 0 or flatind>=grid.n_bins:
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
							   +'EsagridFileBinGroup {}'.format(self.groupname)))

	def check_bin_group_metadata(self,h5group,fix=False,raise_error=False):
		#Check that this group matches this object
		self._check_bin_group_name(h5group)

		#Check for old/missing metadata (e.g. flatind not an integer)
		for attrname,attrval in self.esagrid_bin.items():
			if attrname in h5group.attrs:
				if h5group.attrs[attrname]!=attrval:
					errstr = ('Group:{}\n'.format(h5group.name)
							  +'Incorrect Attribute {}'.format(attrname)
						      +'{}!={}'.format(attrval,h5group.attrs[attrname]))
					
					if raise_error:
						raise BinGroupMetadataError(errstr)
					else:
						print(errstr)

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

	def append_from_other(self,h5f,other_h5f):
		self_h5group = self.get_h5group(h5f)
		other_h5group = other_h5f[self.groupname]
		self.check_bin_group_metadata(other_h5group,fix=False,raise_error=True)
		for h5dsname,h5ds in other_h5group.items():
			if not isinstance(h5ds,h5py.Dataset):
				print('Will not copy {}, not a HDF5 dataset'.format(h5dsname))

			additional_attrs = {key:val for key,val in h5ds.attrs.items()}
			data = h5ds[:]
			self.store(h5f,h5dsname,data,additional_attrs=additional_attrs)

	def store(self,h5f,jd,data,additional_attrs=None,silent=False):
		"""Store an array of data taken at a particular time. Time
		is assumed to be specified as a floating point number (Julian date)
		"""
		h5grp = self.get_h5group(h5f)

		if isinstance(jd,np.ndarray):
			#If time is an array, use the first value in the bin
			#as the hdf5 dataset name
			h5datasetnm = str(jd.flatten()[0])
		else:
			#If time is not an array, just use it's string
			#version as the dataset name
			h5datasetnm = str(jd)

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
			print("Added %d points to %s" % (data.size,
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

class DefaultEsagrid(Esagrid):
	"""Default settings of a 3 degrees latitude per band, 3 cap bins, with
	localtime as the azimuthal coordinate"""
	def __init__(self,delta_lat=3,n_cap_bins=3,azi_coord='lt'):
		Esagrid.__init__(self,delta_lat,n_cap_bins=n_cap_bins,azi_coord=azi_coord)


class EsagridFile(object):
	"""Class representing HDF5 file which contains one Group for each (populated) bin

	PARAMETERS
	----------
	hdf5_filenm : str
		The filename of the hdf5 file to store to. If this is an existing file
		and clobber == False, will use the stored metadata in the file to
		create the appropriate esagrid and you can continue adding to the file
		or process results from it
	grid : esabin.esagrid.ConstantLatitudeSpacingGrid, optional
		The grid of bins to bin into. If it is None (default), a default grid
		with delta_lat = 3 and n_cap_bins = 3 and azi_coord = 'lt' is used
	hdf5_local_dir : str, optional
		A valid local path at which the hdf5 files will be created
	clobber : bool, optional
		If True, will delete and overwrite the HDF5 file specified as 
		os.path.join(hdf5_local_dir,hdf5_filenm) if it exists.
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
		with h5py.File(self.h5fn,'r') as h5f:
			for groupname in h5f:
				try:
					bingroup = EsagridFileBinGroup.from_groupname(self.grid,
																	groupname)
				except InvalidBinGroupNameError:
					print(('Could not create EsagridFileBinGroup'
						   +'from h5 group {}'.format(groupname)))
					continue
				self._bingroups[bingroup['flatind']]=bingroup

	@classmethod
	def copy_of_existing(cls,source_esagrid_file,destination_h5fn):
		"""Generate a new EsagridFile that contains all of the data from
		an existing one under a new filename"""
		if os.path.exists(destination_h5fn):
			raise IOError('Destination file {} exists!'.format(destination_h5fn))

		shutil.copyfile(source_esagrid_file.h5fn,destination_h5fn)
		dest_hdf5_local_dir,dest_hdf5_filename = os.path.split(destination_h5fn)
		return EsagridFile(dest_hdf5_filename,hdf5_local_dir=dest_hdf5_local_dir)

	def append_existing(self,existing_esagrid_file):
		"""Copy all bin data from another EsagridFile instance into this one"""
		with h5py.File(self.h5fn,'a') as h5f:
			with h5py.File(existing_esagrid_file.h5fn,'r') as other_h5f: 
				for flatind in existing_esagrid_file:
					if flatind not in self:
						self[flatind]=EsagridFileBinGroup(self.grid,flatind)
					self[flatind].append_from_other(h5f,other_h5f)

	def __getitem__(self,flatind):
		return self._bingroups[flatind]

	def __setitem__(self,flatind,esagrid_file_bingroup):
		if not isinstance(esagrid_file_bingroup,EsagridFileBinGroup):
			raise TypeError(('{}'.format(esagrid_file_bingroup)
							+' is not an EsagridFileBinGroup'))
		self._bingroups[flatind] = esagrid_file_bingroup

	def __contains__(self,flatind):
		return flatind in self._bingroups

	def items(self):
		return self._bingroups.items()

	def __iter__(self):
		for flatind in self._bingroups:
			yield flatind

	def write_grid_metadata(self):
		with h5py.File(self.h5fn,'a') as h5f:
			h5f.attrs['delta_lat'] = self.grid.delta_lat
			h5f.attrs['n_cap_bins'] = self.grid.n_cap_bins
			h5f.attrs['azi_coord'] = self.grid.azi_coord

	def create_grid_from_metadata(self):
		with h5py.File(self.h5fn,'r') as h5f:
			delta_lat = h5f.attrs['delta_lat']
			try:
				azi_coord = str(h5f.attrs['azi_coord'],'utf8')
			except:
				azi_coord = h5f.attrs['azi_coord']

			if 'n_cap_bins' in h5f.attrs:
				n_cap_bins = h5f.attrs['n_cap_bins']
				return Esagrid(delta_lat,n_cap_bins=n_cap_bins,azi_coord=azi_coord)
			elif 'delta_azi' in h5f.attrs:
				delta_azi = h5f.attrs['delta_azi']
				return ConstantAzimuthalSpacingGrid(delta_lat,delta_azi,azi_coord)
			else:
				raise ValueError(('Missing H5 attribute; cannot specify grid'
				                  +'\ndid not find either "n_cap_bins" (esagrid)'
				                  +'\n or "delta_azi" (constant azi spacing grid)'
				                  +'\nin attrs: {}'.format(h5f.attrs)))

	def bin_and_store(self,t,lat,lonorlt,data,silent=False,additional_attrs=None):
		"""Store data into HDF5 file bin groups. 

		All arrays must be shape (n,) or shape (n,1) or shape (1,n).

		PARAMETERS
		----------

		t : np.ndarray
			Array of times (any float numeric representation)
		lat : np.ndarray
			Array of latitudes
		lonorlt : np.ndarray
			Array of longitudes or local times
		data : np.ndarray
			Array of data to bin
		silent : bool,optional
			Do not print status messages (default False)
		additional_attrs : dict,optional
			A dictionary of additional information which will be stored as
			HDF5 attributes attached to any Datasets created by this function
			call. Keys will be used as attribute names. Attribute values will
			be stored as string representations of dictionary values (str(val)).
		"""
		latbands,lonbins,flatinds = self.grid.whichbin(lat,lonorlt)

		with h5py.File(self.h5fn,'a') as h5f:
			for bin_ind in np.unique(flatinds):

				in_bin = flatinds == bin_ind

				if bin_ind not in self:
					self[bin_ind] = EsagridFileBinGroup(self.grid,bin_ind)

				self[bin_ind].store(h5f,
									t[in_bin].flatten(),
									data[in_bin].flatten(),
									additional_attrs=additional_attrs,
									silent=silent)

	def _dataset_passes_attr_filters(self,dataset,attr_filters,default_result=True):
		"""
		Filtering whether to include a specific dataset in a bin_stats
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
		"""Apply some function to the contents of each bin

		PARAMETERS
		----------
		statfun : callable or list, optional
			The function which will be called. Can also be a list
			of multiple callables (in which case a dict keyed with
			str(callable) is returned). It is MUCH MORE EFFICIENT to
			do this than call bin_stats multiple times.
		statfunname : str or list, optional
			List of custom keys (use if statfun is a list of callables)
		center_or_edges : {'center','edges'},optional
			Return the bin center or edges
		minlat : float,optional
			Minimum absolute latitude (default: 50.) below 
			which bins are ignored.
		silent : bool,optional
			Do not print status messages to stdout, default False
		force_recompute : bool,optional
			If a particular statfun callable has be evaluated before,
			do not return a cached result. Default False.
		write_to_h5 : bool, optional
			Cache the results for particular statfun callable in the HDF5.
			Default True. If statfunname is defined, cached result will
			be stored under that name (identity of callable is not checked)
		attr_filters : dict, optional
			Provides optional filtering of the data fed to the callable using
			HDF5 attributes of the bin Datasets (additional_attrs) 
			Key should be the HDF5 Dataset attribute name, 
			value should be a callable which takes attribute value 
			and returns True or False.

		RETURNS
		-------
		binlats : np.ndarray
			Bin latitudes (center or edges)
		binlonorlt : np.ndarray
			Bin longitudes or local times (center or edges)
		binstats : np.ndarray or dict
			If statfun is a single callable, will return an array, the
			result of evaluating statfun. If statfun is a list, 
			this will be a dict of arrays, keyed with either the string
			representation of each statfun or statfunnames if they are
			defined.
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

		with h5py.File(self.h5fn,'a') as h5f:
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
						flatind = EsagridFileBinGroup._flatind_from_group_name(groupnm)
					except InvalidBinGroupNameError:
						print('{} is not a bin group, skipping'.format(groupnm))
						continue

					esagrid_file_bingroup = EsagridFileBinGroup(self.grid,flatind)
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
						is_okay = self._dataset_passes_attr_filters(dataset,
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

