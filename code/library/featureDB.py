############################################
#Measurements of features on shear catalogs#
############################################

from __future__ import division,print_function,with_statement
import sys,os
from itertools import product

from lenstools.catalog import ShearCatalog
from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.database import Database

import numpy as np
import astropy.units as u
import astropy.table as tbl


#############################################
#Measure features in a partcular realization#
#############################################

def process_realization(realization,db_type,map_specs,sub_catalog,measurer,**kwargs):

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
	position_files = [ "/global/homes/a/apetri/LSST100Parameters/data/positions_bin{0}.fits".format(n) for n in range(1,map_specs["nzbins"]+1) ]
	shear_files = [ os.path.join(sub_catalog.storage_subdir,"WLshear_positions_bin{0}_{1:04d}r.fits".format(n,realization)) for n in range(1,map_specs["nzbins"]+1) ]

	#Construct the maps
	maps = db_type.make_maps(shear_files,position_files,npixel=map_specs["npixel"],smooth=map_specs["smooth"],fov=map_specs["fov"],add_noise=map_specs["add_noise"])

	#Measure the features
	ensemble_realization = measurer(maps,**kwargs)

	#Add the realization label
	ensemble_realization["realization"] = realization

	#Return 
	return ensemble_realization

#Aggregate results of computation
def _assemble(ens_list):
	return Ensemble.concat(ens_list,axis=0,ignore_index=True)

################################
#####FeatureDatabase class######
################################

class FeatureDatabase(Database):

	#Global options
	map_specs = {
	"nzbins" : 5,
	"npixel" : 512,
	"smooth" : 0.5*u.arcmin,
	"fov" : 3.5*u.deg,
	"add_noise" : False
	}

	def __init__(self,name,**kwargs):
		super(FeatureDatabase,self).__init__(name)
		for key in kwargs.keys():
			self.map_specs[key] = kwargs[key]

	###########################################################################################
	#Process all the realizations in a particular sub-catalog; add the results to the database#
	###########################################################################################

	def add_features(self,table_name,sub_catalog,measurer,extra_columns=None,pool=None,**kwargs):

		"""

		:param table_name: name of the SQL table to add the feature to
		:type table_name: str.

		:param sub_catalog: sub catalog that the realization belongs to
		:type sub_catalog: SimulationSubCatalog

		:param measurer: gets called on the list of convergence maps in the particular realization, must return an Ensemble
		:type measurer: callable

		:param extra_columns: dictionary whose keys are the names of the extra columns to insert in the database, and whose values are the column values for the sub_catalog
		:type extra_columns: dict.

		:param kwargs: passed to the measurer
		:type kwargs: dict.

		:param pool: MPIPool to spread the sub_catalog computations onto
		:type pool: MPIPool

		"""

		#First and last realization to process in the sub_catalog
		first_realization = sub_catalog.first_realization
		last_realization = sub_catalog.last_realization

		#Compute Ensemble of realizations
		ensemble_sub_catalog = Ensemble.compute(range(first_realization,last_realization+1),callback_loader=process_realization,assemble=_assemble,pool=pool,map_specs=self.map_specs,db_type=self.__class__,sub_catalog=sub_catalog,measurer=measurer,**kwargs)

		#Add the cosmological parameters as additional columns
		ensemble_sub_catalog["Om"] = sub_catalog.cosmology.Om0
		ensemble_sub_catalog["w"] = sub_catalog.cosmology.w0
		ensemble_sub_catalog["sigma8"] = sub_catalog.cosmology.sigma8

		#Add the extra columns
		if extra_columns is not None:
			for key in extra_columns.keys():
				
				if key in ensemble_sub_catalog.columns:
					raise ValueError("Column {0} already present in database!".format(key))
				
				ensemble_sub_catalog[key] = extra_columns[key]

		#Insert into the database
		self.insert(ensemble_sub_catalog,table_name)


	###############################################
	#Create convergence maps out of shear catalogs#
	###############################################
	
	@staticmethod
	def make_maps(shear_files,position_files,npixel=512,smooth=0.5*u.arcmin,fov=3.5*u.deg,add_noise=False):

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

	##############################
	#Principal component analysis#
	##############################

	@staticmethod
	def _sql_sub_indices(sub_indices):
		return " AND ".join(["{0}={1}".format(key,sub_indices[key]) for key in sub_indices.keys() if sub_indices[key] is not None])

	@staticmethod
	def _suppress_indices(sub_indices):
		return [key for key in sub_indices.keys() if sub_indices[key] is None]

	#PCA on the cosmological models, for each realization
	def pca_models(self,features,sub_indices,realizations,num_modes,db_name,table_name,base_table_name="features",location=None,scale=None):
		
		"""
		Perform Principal Component Analysis for each realization contained in the database

		:param features: list of features to consider
		:type features: list. 

		:param sub_indices: dictionary with the sub indices to select (typically redshift indices); the keys are the column names and the values are the corresponding column values
		:type sub_indices: dict.

		:param realizations: repeat the PCA for these realizations
		:type realizations: list.

		:param num_modes: number of PCA modes to include in the PCA database
		:type num_modes: int.

		:param db_name: name of the Database on which to write the PCA information
		:type db_name: str.

		:param table_name: database table name on which to write the PCA information
		:type table_name: str.

		:param base_table_name: table name to query in the current Database
		:type base_table_name: str.

		:param location: perform the PCA with respect to this location
		:type location: Series

		:param scale: scale the features with these weights before performing the PCA
		:type scale: Series

		"""

		#We use a context manager to populate the Database that contains the PCA information
		with self.__class__(db_name) as db:

			#Cycle over realizations
			for realization in realizations:

				#Build the query
				query = "SELECT * FROM {0} WHERE realization={0}"
				sub_indices_query = self._sql_sub_indices(sub_indices)
				if len(sub_indices_query):
					query += " AND " + sub_indices_query

				#Query the Database
				print("[+] Executing SQL query: {0}".format(query))
				ens = self.query(query)

				#Contract the sub_indices
				suppress = self._suppress_indices(sub_indices)
				if len(suppress):
					labels,ens = ens.suppress_indices(by=["model"],suppress=suppress,columns=features)
					features = list(product(features,range(len(labels))))

				#Safety check: there should be exactly one row per model at this point
				assert len(ens)==len(ens["model"].drop_duplicates()),"There should be exactly one line per model in the Ensemble before performing the PCA!"

				#Perform the PCA
				pca = ens[features].principalComponents(location=location,scale=scale)
				mode_directions = pca.directions.head(num_modes)
				mode_directions["eigenvalue"] = pca.eigenvalues[:num_modes]
				mode_directions["realization"] = realization

				#Fill in the additional indices
				for key in sub_indices.keys():
					if sub_indices[key] is not None:
						mode_directions[key] = sub_indices[key]

				#Insert in the Database
				db.insert(mode_directions,table_name)

	########################################################################################################################################################################################

	def pca_sub_catalog(self):
		pass
	
