import os
import argparse

import numpy as np

from lenstools.utils.decorators import Parallelize
from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.constraints import Emulator

from featureDB import FeatureDatabase
from defaults import settings as default_settings

########################################
#######Measure features driver##########
########################################

@Parallelize.masterworker
def measure(batch,cosmo_id,catalog_names,model_n,db_name,table_names,measurer,pool,**kwargs):

	if isinstance(catalog_names,str):
		catalog_names = [catalog_names]

	if isinstance(table_names,str):
		table_names = [table_names]*len(catalog_names)

	#Populate database
	db_full_name = os.path.join(batch.environment.storage,db_name)
	with FeatureDatabase(db_full_name) as db:
			
		#Handle on the model 
		model = batch.getModel(cosmo_id)
			
		#Process sub catalogs
		for nc,catalog_name in enumerate(catalog_names):

			#Log to user
			print("[+] Populating table '{0}' of database {1}...".format(table_names[nc],db_full_name))

			for s,sc in enumerate(model.getCollection("512b260").getCatalog(catalog_name).subcatalogs):
				print("[+] Processing model {0}, catalog {1}, sub-catalog {2}...".format(model_n,catalog_name,s+1))
				db.add_features(table_names[nc],sc,measurer=measurer,extra_columns={"model":model_n,"sub_catalog":s+1},pool=pool,**kwargs)


################################################
#######Cosmological constraints driver##########
################################################

def cosmo_constraints(batch,specs,settings=default_settings):

	####################################################################################################################
	##Specs should be a dictionary with all the information needed to combine the features and compute the constraints##
	####################################################################################################################

	#Placeholders for emulator, data and covariance
	emulator = Ensemble(columns=["model"])
	covariance = Ensemble(columns=["realization"])
	data = Ensemble(columns=["realization"])

	#First read in which kind of features are we combining
	for feature in specs["features"]:

		#Query the database
		feature_dbfile = os.path.join(batch.storage,getattr(settings,feature).dbname)
		print("")
		print("[+] Reading {0} from {1}".format(feature,feature_dbfile))

		with FeatureDatabase(feature_dbfile) as db:

			######################
			#Read in the emulator#
			######################

			print("[+] Reading emulator from table {0}".format(getattr(settings,feature).emulator_table))
			sql_query = getattr(settings,feature).emulator_query(feature_filter=specs[feature]["feature_filter"],redshift_filter=specs[feature]["redshift_filter"])
			print("[+] SQL: {0}".format(sql_query))
			models = db.read_table(getattr(settings,feature).models_table)
			query_results = db.query(sql_query)
			
			#Suppress redshift indices
			l,query_results = query_results.suppress_indices(by=["model"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Emulators are merged on model
			emulator = Ensemble.merge(emulator,query_results,on=["model"],how="outer")

			########################
			#Read in the covariance#
			########################

			print("[+] Reading covariance from table {0}".format(getattr(settings,feature).covariance_table))
			sql_query = getattr(settings,feature).covariance_query(feature_filter=specs[feature]["feature_filter"],redshift_filter=specs[feature]["redshift_filter"],realization_filter=specs[feature]["realization_filter"])
			print("[+] SQL: {0}".format(sql_query))
			query_results = db.query(sql_query)

			#Suppress redshift indices
			l,query_results = query_results.suppress_indices(by=["realization"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Covariances are merged on realization
			covariance = Ensemble.merge(covariance,query_results,on=["realization"],how="outer")

			##################
			#Read in the data#
			##################

			print("[+] Reading data to fit from table {0}".format(getattr(settings,feature).data_table))
			sql_query = getattr(settings,feature).data_query(feature_filter=specs[feature]["feature_filter"],redshift_filter=specs[feature]["redshift_filter"],realization_filter=specs[feature]["realization_filter"])
			print("[+] SQL: {0}".format(sql_query))
			query_results = db.query(sql_query)

			#Suppress redshift indices
			l,query_results = query_results.suppress_indices(by=["realization"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Data is merged on realization
			data = Ensemble.merge(data,query_results,on=["realization"],how="outer")

	#Attach the models to the emulator
	emulator = Ensemble.merge(models,emulator,on=["model"])

	######################
	##Calculations begin##
	######################

	emulator.save("emulator.pkl")
	covariance.save("covariance.pkl")
	data.save("data.pkl")






