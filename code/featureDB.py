############################################
#Measurements of features on shear catalogs#
############################################

from __future__ import division,print_function,with_statement
import sys,os

from lenstools.catalog import ShearCatalog
from lenstools.statistics.ensemble import Ensemble

import numpy as np
import astropy.units as u
import astropy.table as tbl

from sqlalchemy import create_engine

class FeatureDatabase(object):

	#Global options
	nzbins = 5
	npixel = 512
	smooth = 0.5*u.arcmin
	fov = 3.5*u.deg
	add_noise = False

	#Create a connection to a database
	def __init__(self,name):
		self.connection = create_engine("sqlite:///"+name)

	#For context manager
	def __enter__(self):
		return self

	def __exit__(self,type,value,tb):
		self.connection.dispose()

	#Insert records in the database
	def insert(self,df,table_name="data"):

		"""
		:param df: records to insert in the database, in Ensemble (or pandas DataFrame) format
		:type df: Ensemble

		"""

		df.to_sql(table_name,self.connection,if_exists="append",index=False)

	###########################################################################################
	#Process all the realizations in a particular sub-catalog; add the results to the database#
	###########################################################################################

	def add_features(self,table_name,sub_catalog,measurer,pool=None,**kwargs):

		"""

		:param table_name: name of the SQL table to add the feature to
		:type table_name: str.

		:param sub_catalog: sub catalog that the realization belongs to
		:type sub_catalog: SimulationSubCatalog

		:param measurer: gets called on the list of convergence maps in the particular realization, must return an Ensemble
		:type measurer: callable

		:param kwargs: passed to the measurer
		:type kwargs: dict.

		:param pool: MPIPool to spread the sub_catalog computations onto
		:type pool: MPIPool

		"""

		#First and last realization to process in the sub_catalog
		first_realization = sub_catalog.first_realization
		last_realization = sub_catalog.last_realization

		#Compute Ensemble of realizations
		ensemble_sub_catalog = Ensemble.compute(range(first_realization,last_realization+1),callback_loader=self.process_realization,assemble=self._assemble,pool=pool,sub_catalog=sub_catalog,measurer=measurer,**kwargs)

		#Add the cosmological parameters as additional columns
		ensemble_sub_catalog["Om"] = sub_catalog.cosmology.Om0
		ensemble_sub_catalog["w"] = sub_catalog.cosmology.w0
		ensemble_sub_catalog["sigma8"] = sub_catalog.cosmology.sigma8

		#Insert into the database
		self.insert(ensemble_sub_catalog,table_name)


	@staticmethod
	def _assemble(ens_list):
		return Ensemble.concat(ens_list,axis=0,ignore_index=True)

	###############################################
	#Create convergence maps out of shear catalogs#
	###############################################
	
	@staticmethod
	def make_maps(shear_files,position_files,npixel=512,smooth=0.5*u.arcmin,fov=3.5*u.deg):

		"""
		:param shear_files: list of files that contain the shear catalogs
		:type shear_files: list.

		:param position_files: list of files that contain positions and redshift of the galaxies in the shear files
		:type position_files: list.

		:returns: list of ConvergenceMap

		"""

		#Safety checks
		assert len(shear_files)==len(position_files)

		#Read in all the shear catalogs, and convert to convergence maps with E/B mode decomposition
		convergence_maps = list()
		for n,shear_file in enumerate(shear_files):
			full_catalog = tbl.hstack((ShearCatalog.read(position_files[n]),ShearCatalog.read(shear_file)))
			convergence_maps.append(full_catalog.toMap(map_size=fov,npixel=npixel,smooth=smooth).convergence())

		#Return
		return convergence_maps

	#############################################
	#Measure features in a partcular realization#
	#############################################

	@classmethod
	def process_realization(cls,realization,sub_catalog,measurer,**kwargs):

		"""
		:param realization: number of the realization to process
		:type realization: int.

		:param sub_catalog: sub catalog that the realization belongs to
		:type sub_catalog: SimulationSubCatalog

		:param measurer: gets called on the list of convergence maps in the particular realization, must return an Ensemble
		:type measurer: callable

		:param kwargs: passed to the measurer
		:type kwargs: dict.

		:returns: Ensemble

		"""

		#Construct file names of shear and position catalogs
		position_files = [ "../data/positions_bin{0}.fits".format(n) for n in range(1,cls.nzbins+1) ]
		shear_files = [ os.path.join(sub_catalog.storage_subdir,"WLshear_positions_bin{0}_{1:04d}r.fits".format(n,realization)) for n in range(1,cls.nzbins+1) ]

		#Construct the maps
		maps = cls.make_maps(shear_files,position_files,npixel=cls.npixel,smooth=cls.smooth,fov=cls.fov)

		#Measure the features
		ensemble_realization = measurer(maps,**kwargs)

		#Add the realization label
		ensemble_realization["realization"] = realization

		#Return 
		return ensemble_realization
	
	##########################################################################################################################################################################################

	##########################################################
	#Measure the cross spectrum of a list of convergence maps#
	##########################################################

	@staticmethod
	def cross_power(maps,l_edges,indices):

		"""

		:param maps: list of convergence maps (one for each redshift bin)
		:type maps: list.

		:param l_edges: multipoles
		:type l_edges: array

		:param indices: pairs of indices; each pair corresponds to a pair of redshift bins to cross correlate
		:type indices: list of tuples

		:returns: Ensemble

		"""

		#Allocate memory
		cross_power_array = np.zeros((len(indices),len(l_edges)-1))
		
		#Measure the auto and cross power spectrum
		for n,(i,j) in enumerate(indices):
			ell,cross_power_array[n] = maps[i].cross(maps[j],statistic="power_spectrum",l_edges=l_edges)

		#Build the Ensemble
		columns = [ "l{0}".format(n) for n in range(1,len(ell)+1) ]
		ensemble = Ensemble(cross_power_array,columns=columns)

		#Add the indices labels
		ensemble["b1"],ensemble["b2"] = zip(*indices)

		#Return
		return ensemble


