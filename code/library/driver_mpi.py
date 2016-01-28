import os
import argparse
import itertools

import numpy as np
import pandas as pd

from lenstools.simulations.logs import logdriver
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
			logdriver.info("Populating table '{0}' of database {1}...".format(table_names[nc],db_full_name))

			for s,sc in enumerate(model.getCollection("512b260").getCatalog(catalog_name).subcatalogs):
				logdriver.info("Processing model {0}, catalog {1}, sub-catalog {2}...".format(model_n,catalog_name,s+1))
				db.add_features(table_names[nc],sc,measurer=measurer,extra_columns={"model":model_n,"sub_catalog":s+1},pool=pool,**kwargs)


################################################
#######Cosmological constraints driver##########
################################################

def cosmo_constraints(batch,specs,settings=default_settings,verbose=False):

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
		logdriver.info("Reading {0} from {1}".format(feature,feature_dbfile))

		with FeatureDatabase(feature_dbfile) as db:

			######################
			#Read in the emulator#
			######################

			logdriver.info("Reading emulator from table {0}".format(getattr(settings,feature).emulator_table))
			sql_query = getattr(settings,feature).emulator_query(feature_filter=specs[feature]["feature_filter"],redshift_filter=specs[feature]["redshift_filter"])
			logdriver.info("SQL: {0}".format(sql_query))
			models = db.read_table(getattr(settings,feature).model_table)
			query_results = db.query(sql_query)
			
			#Suppress redshift indices
			logdriver.info("Suppressing redshift indices...")
			l,query_results = query_results.suppress_indices(by=["model"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Emulators are merged on model
			logdriver.info("Merging to master emulator database...")
			emulator = Ensemble.merge(emulator,query_results,on=["model"],how="outer")

			########################
			#Read in the covariance#
			########################

			logdriver.info("Reading covariance from table {0}".format(getattr(settings,feature).covariance_table))
			sql_query = getattr(settings,feature).covariance_query(feature_filter=specs[feature]["feature_filter"],redshift_filter=specs[feature]["redshift_filter"],realization_filter=specs[feature]["realization_filter"])
			logdriver.info("SQL: {0}".format(sql_query))
			query_results = db.query(sql_query)

			#Suppress redshift indices
			logdriver.info("Suppressing redshift indices...")
			l,query_results = query_results.suppress_indices(by=["realization"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Covariances are merged on realization
			logdriver.info("Merging to master covariance database...")
			covariance = Ensemble.merge(covariance,query_results,on=["realization"],how="outer")

			##################
			#Read in the data#
			##################

			logdriver.info("Reading data to fit from table {0}".format(getattr(settings,feature).data_table))
			sql_query = getattr(settings,feature).data_query(feature_filter=specs[feature]["feature_filter"],redshift_filter=specs[feature]["redshift_filter"],realization_filter=specs[feature]["realization_filter"])
			logdriver.info("SQL: {0}".format(sql_query))
			query_results = db.query(sql_query)

			#Suppress redshift indices
			logdriver.info("Suppressing redshift indices...")
			l,query_results = query_results.suppress_indices(by=["realization"],suppress=getattr(settings,feature).redshift_labels,columns=getattr(settings,feature).feature_labels)
			
			#Data is merged on realization
			logdriver.info("Merging to master data database...")
			data = Ensemble.merge(data,query_results,on=["realization"],how="outer")

	print("")

	#Attach the models to the emulator
	logdriver.info("Attaching cosmological parameter values...")
	emulator = Ensemble.merge(models,emulator,on=["model"])

	#Cast the emulator into an Emulator instance
	pnames = filter(lambda n:n!="model",models.columns)
	parameters = emulator[pnames]
	parameters.add_name("parameters")
	
	for column in pnames+["model"]:
		emulator.pop(column)
	
	emulator.add_name("features")
	emulator = Emulator.from_features(emulator,parameters)

	#Pop 'realization' columns that are not needed anymore, add names to data and covariance to match labels
	for df in [data,covariance]:
		df.pop("realization")
		df.add_name("features")

	###########################
	##PCA projections (maybe)##
	###########################

	if "pca_components" in specs:

		#Log the initial size of the feature vector
		logdriver.info("Initial size of the feature vector: {0}".format(covariance.shape[1]))

		#Use the emulator to figure out the PCA components
		logdriver.info("Computing principal components...")
		pca = emulator[["features"]].principalComponents(scale=emulator[["features"]].mean())

		#Transform the emulator,data and covariance with a PCA projection
		logdriver.info("Projecting emulator on principal components...")
		emulator = emulator.refeaturize(pca.transform,method="apply_whole")

		logdriver.info("Projecting covariance ensemble on principal components...")
		covariance = pca.transform(covariance)

		logdriver.info("Projecting data ensemble on principal components...")
		data = pca.transform(data)

		#Add names to data and covariance to match labels
		for df in [data,covariance]:
			df.add_name("features")

	#############
	#Constraints#
	#############

	#Column names
	pcov_columns = ["{0}-{1}".format(i,j) for i,j in itertools.product(pnames,pnames)]

	#Constraints database
	outdbname = os.path.join(batch.home,"data",specs["dbname"])
	logdriver.info("Saving constraints to {0}, table '{1}'".format(outdbname,specs["table_name"]))

	#Number of PCA components to process
	if "pca_components" in specs:
		pca_components = specs["pca_components"]
	else:
		pca_components = [None]

	with FeatureDatabase(outdbname) as db:

		for nc in pca_components:

			print("")

			#Build feature label that enters in database
			feature_label = specs["feature_label_format"].format(specs["feature_label_root"],nc,specs["realizations_for_covariance"],specs["realizations_for_data"])

			#Process different numbers of PCA components
			if nc is not None:
				
				logdriver.info("Processing {0} PCA components; feature label '{1}'".format(nc,feature_label))

				#Build names of column subset
				columns_nc = [("features",n) for n in range(nc)]
				emulator_nc = [("parameters",p) for p in pnames] + columns_nc

				#Truncate emulator,covariance and data to the appropriate number of principal components
				emulator_pca = emulator[emulator_nc].copy()
				covariance_pca = covariance[columns_nc]
				data_pca = data[columns_nc]
			
			else:

				#If PCA is switched off, there is nothing to do
				logdriver.info("Processing whole feature vector without PCA")
				emulator_pca = emulator
				covariance_pca = covariance
				data_pca = data

			#If verbosity is on, log the full feature vector to the user
			if verbose:
				logdriver.debug("Feature vector is: {0}".format(",".join([str(c[-1]) for c in covariance_pca.columns])))

			#Train the emulator with a multiquadric kernel
			logdriver.info("Training emulator with multiquadric kernel...")
			emulator_pca.train(method=multiquadric)

			#Approximate the emulator linearly around the fiducial parameters to get a FisherAnalysis instance
			logdriver.info("Approximating the emulator linearly around ({0})=({1}), derivative precision={2:.2f}...".format(",".join(pnames),",".join(["{0:.2f}".format(p) for p in settings.fiducial_parameters]),settings.derivative_precision))
			fisher = emulator_pca.approximate_linear(settings.fiducial_parameters,settings.derivative_precision)

			#Compute the parameter covariance matrix correcting for the inverse covariance bias
			logdriver.info("Computing the {0}x{0} feature covariance matrix, Nreal={1}, NLSST={2}...".format(covariance_pca.shape[1],specs["realizations_for_covariance"],settings.covariance_to_lsst))
			feature_covariance = covariance_pca.head(specs["realizations_for_covariance"]).cov() / settings.covariance_to_lsst

			logdriver.info("Computing {0}x{0} parameter covariance matrix, Nbins={1}...".format(len(pnames),covariance_pca.shape[1]))
			parameter_covariance = fisher.parameter_covariance(feature_covariance,correct=specs["realizations_for_covariance"])

			################################
			#Output to constraints database#
			################################

			#Format the row to insert
			row = pd.Series(parameter_covariance.values.flatten(),index=pcov_columns)
			row["bins"] = covariance_pca.shape[1]
			row["feature_label"] = feature_label

			#Insert the row
			db.insert(pd.DataFrame(row).T,specs["table_name"])






