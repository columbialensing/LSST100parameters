import os
import argparse

import numpy as np
from lenstools.utils.decorators import Parallelize
from library.featureDB import FeatureDatabase

########################################
#######Measure features driver##########
########################################

@Parallelize.masterworker
def measure(batch,cosmo_id,model_n,db_name,table_name,measurer,pool,**kwargs):

	#Populate database
	db_full_name = os.path.join(batch.environment.storage,db_name)
	with FeatureDatabase(db_full_name) as db:

		print("[+] Populating table '{0}' of database {1}...".format(table_name,db_full_name))
			
		#Handle on the model 
		model = batch.getModel(cosmo_id)
			
		#Process sub catalogs
		for s,sc in enumerate(model.getCollection("512b260").getCatalog("Shear").subcatalogs):
			print("[+] Processing model {0}, sub-catalog {1}...".format(model_n,s+1))
			db.add_features(table_name,sc,measurer=measurer,extra_columns={"model":model_n,"sub_catalog":s+1},pool=pool,**kwargs)

###################################################################
#######Compute scores of a grid of parameter combinations##########
###################################################################

def chi2score(emulator,parameters,data,data_covariance,nchunks,pool):

	#Score the data on each of the parameter combinations provided
	scores = emulator.score(parameters,data,features_covariance=data_covariance,split_chunks=nchunks,pool=pool)

	#Pop the parameter columns, compute the likelihoods out of the chi2
	for p in parameters.columns:
		scores.pop(p)

	return scores,scores.apply(lambda c:np.exp(-0.5*c),axis=0)

@Parallelize.masterworker
def chi2database(db_name,parameters,specs,pool=None,nchunks=None):

	#Each processor should have the same exact workload
	if nchunks is not None:
		assert not len(parameters)%nchunks

	#Database context manager
	print("[+] Populating score database {0}...".format(db_name))
	with FeatureDatabase(db_name) as db:

		#Repeat the scoring for each key in the specs dictionary
		for feature_type in specs.keys():

			#Log
			print("[+] Processing feature_type: {0} ({1} parameter combinations)...".format(feature_type,len(parameters)))
			
			#Score
			chi2,likelihood = chi2score(emulator=specs[feature_type]["emulator"],parameters=parameters,data=specs[feature_type]["data"],data_covariance=specs[feature_type]["data_covariance"],nchunks=nchunks,pool=pool)
			assert (chi2.columns==[feature_type]).all()

			#Add to the database
			db_chunk = parameters.copy()
			db_chunk["feature_type"] = feature_type
			db_chunk["chi2"] = chi2
			db_chunk["likelihood"] = likelihood

			db.insert(db_chunk,"scores")
