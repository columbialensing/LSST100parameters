import os
import argparse

from lenstools.utils.decorators import Parallelize
from lenstools.pipeline import SimulationBatch
from lenstools.pipeline.settings import EnvironmentSettings

from library.featureDB import FeatureDatabase

############################
#######Main driver##########
############################

@Parallelize.masterworker
def main(db_name,measurer,pool,**kwargs):

	#parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--environment",dest="environment",help="Environment options file")
	parser.add_argument("-c","--config",dest="config",help="Configuration file")
	parser.add_argument("id",nargs="*")
	cmd_args = parser.parse_args()

	#Handle on the current batch
	batch = SimulationBatch(EnvironmentSettings.read(cmd_args.environment))

	#Populate database
	with FeatureDatabase(os.path.join(batch.environment.storage,db_name)) as db:
		
		for model_id in cmd_args.id:
			
			#Handle on the model
			cosmo_id,n = model_id.split("|") 
			model = batch.getModel(cosmo_id)
			
			#Process sub catalogs
			for s,sc in enumerate(model.getCollection("512b260").getCatalog("Shear").subcatalogs):
				print("[+] Processing model {0}, sub-catalog {1}...".format(int(n),s+1))
				db.add_features("features",sc,measurer=measurer,extra_columns={"model":int(n),"sub_catalog":s+1},pool=pool,**kwargs)