import os
import argparse

import numpy as np
from lenstools.utils.decorators import Parallelize
from library.featureDB import FeatureDatabase

########################################
#######Measure features driver##########
########################################

@Parallelize.masterworker
def measure(batch,cosmo_id,catalog_name,model_n,db_name,table_name,measurer,pool,**kwargs):

	#Populate database
	db_full_name = os.path.join(batch.environment.storage,db_name)
	with FeatureDatabase(db_full_name) as db:

		print("[+] Populating table '{0}' of database {1}...".format(table_name,db_full_name))
			
		#Handle on the model 
		model = batch.getModel(cosmo_id)
			
		#Process sub catalogs
		for s,sc in enumerate(model.getCollection("512b260").getCatalog(catalog_name).subcatalogs):
			print("[+] Processing model {0}, sub-catalog {1}...".format(model_n,s+1))
			db.add_features(table_name,sc,measurer=measurer,extra_columns={"model":model_n,"sub_catalog":s+1},pool=pool,**kwargs)