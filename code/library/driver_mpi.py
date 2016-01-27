import os
import argparse
import itertools

import numpy as np
import pandas as pd

from lenstools.utils.decorators import Parallelize
from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.constraints import Emulator
from lenstools.statistics.samplers import multiquadric

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
			models = db.read_table(getattr(settings,feature).model_table)
			query_results = db.query(sql_query)
			
			#Suppress redshift indices
			print("[+] Suppressing redshift indices...")
			l,query_results = query_results.suppress_indices(by=["model"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Emulators are merged on model
			print("[+] Merging to master database...")
			emulator = Ensemble.merge(emulator,query_results,on=["model"],how="outer")

			########################
			#Read in the covariance#
			########################

			print("[+] Reading covariance from table {0}".format(getattr(settings,feature).covariance_table))
			sql_query = getattr(settings,feature).covariance_query(feature_filter=specs[feature]["feature_filter"],redshift_filter=specs[feature]["redshift_filter"],realization_filter=specs[feature]["realization_filter"])
			print("[+] SQL: {0}".format(sql_query))
			query_results = db.query(sql_query)

			#Suppress redshift indices
			print("[+] Suppressing redshift indices...")
			l,query_results = query_results.suppress_indices(by=["realization"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Covariances are merged on realization
			print("[+] Merging to master database...")
			covariance = Ensemble.merge(covariance,query_results,on=["realization"],how="outer")

			##################
			#Read in the data#
			##################

			print("[+] Reading data to fit from table {0}".format(getattr(settings,feature).data_table))
			sql_query = getattr(settings,feature).data_query(feature_filter=specs[feature]["feature_filter"],redshift_filter=specs[feature]["redshift_filter"],realization_filter=specs[feature]["realization_filter"])
			print("[+] SQL: {0}".format(sql_query))
			query_results = db.query(sql_query)

			#Suppress redshift indices
			print("[+] Suppressing redshift indices...")
			l,query_results = query_results.suppress_indices(by=["realization"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Data is merged on realization
			print("[+] Merging to master database...")
			data = Ensemble.merge(data,query_results,on=["realization"],how="outer")

	#Attach the models to the emulator
	print("[+] Attaching cosmological parameter values...")
	emulator = Ensemble.merge(models,emulator,on=["model"])

	#Cast the emulator into an Emulator instance
	pnames = filter(lambda n:n!="model",models.columns)
	parameters = emulator[pnames]
	parameters.add_name("parameters")
	
	for column in pnames+["model"]:
		emulator.pop(column)
	
	emulator.add_name("features")
	emulator = Emulator.from_features(emulator,parameters)

	###################
	##PCA projections##
	###################

	#Pop 'realization' columns that are not needed anymore, add names to data and covariance to match labels
	for df in [data,covariance]:
		df.pop("realization")
		df.add_name("features")

	#############
	#Constraints#
	#############

	#Train the emulator with a multiquadric kernel
	print("[+] Training emulator with multiquadric kernel...")
	emulator.train(method=multiquadric)

	#Approximate the emulator linearly around the fiducial parameters to get a FisherAnalysis instance
	print("[+] Approximating the emulator linearly around ({0})=({1}), derivative precision={2:.2f}...".format(",".join(pnames),",".join(["{0:.2f}".format(p) for p in settings.fiducial_parameters]),settings.derivative_precision))
	fisher = emulator.approximate_linear(settings.fiducial_parameters,settings.derivative_precision)

	#Compute the parameter covariance matrix correcting for the inverse covariance bias
	print("[+] Computing the {0}x{0} feature covariance matrix, Nreal={1}...".format(covariance.shape[1],specs["realizations_for_covariance"]))
	feature_covariance = covariance.head(specs["realizations_for_covariance"]).cov()

	print("[+] Computing {0}x{0} parameter covariance matrix, Nbins={1}...".format(len(pnames),covariance.shape[1]))
	parameter_covariance = fisher.parameter_covariance(feature_covariance,correct=specs["realizations_for_covariance"])

	################################
	#Output to constraints database#
	################################

	#Column names
	pcov_columns = ["{0}-{1}".format(i,j) for i,j in itertools.product(pnames,pnames)]

	#Constraints database
	outdbname = os.path.join(batch.home,"data",specs["dbname"])
	print("[+] Saving constraints to {0}, table '{1}'".format(outdbname,specs["table_name"]))

	#Format the row to insert
	row = pd.Series(parameter_covariance.values.flatten(),index=pcov_columns)
	row["bins"] = covariance.shape[1]
	row["feature_label"] = specs["feature_label_root"]

	#Insert the row
	with FeatureDatabase(outdbname) as db:
		db.insert(pd.DataFrame(row).T,specs["table_name"])






