#!/usr/bin/env python

import sys
import json
import argparse

sys.modules["mpi4py"] = None

from lenstools.simulations.logs import logdriver

from library.featureDB import LSSTSimulationBatch,FeatureDatabase
from library.defaults import settings

import pandas as pd

##################
##Main execution##
##################

def main():

	#Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--environment",dest="environment",help="Environment options file")
	parser.add_argument("-d","--database",dest="database",default='{"power_spectrum":"cross_spectra.sqlite"}',help="Dictionary: feature-->database in JSON format")
	parser.add_argument("-m","--models",dest="models",default="1-100",help="models to process")
	parser.add_argument("-p","--parameters",dest="parameters",default="Om,w,sigma8",help="names of the cosmological parameters")
	parser.add_argument("features",nargs="*")
	cmd_args = parser.parse_args()

	#Get a handle on the simulation batch
	logdriver.info("Reading batch locations from {0}".format(cmd_args.environment))
	batch = LSSTSimulationBatch.current(cmd_args.environment)
	logdriver.info("Batch home: {0} Batch storage: {1}".format(batch.home,batch.storage))

	#Read the database dictionary
	feature2database = json.loads(cmd_args.database)

	#Models to process
	try:
		first_model,last_model = [ int(n) for n in cmd_args.models.split("-") ]
		models = range(first_model,last_model+1)
	except ValueError:
		models = [int(cmd_args.models)]


	#Names of the cosmological parameters
	parameter_names = cmd_args.parameters.split(",")
	logdriver.info("Cosmological parameter names: {0}".format(",".join(parameter_names)))

	#Process each feature
	for feature in cmd_args.features:

		dbname = os.path.join(batch.storage,feature2database[feature])
		logdriver.info("Computing means for database at {0}".format(dbname)) 

		#Open the database
		with FeatureDatabase(dbname) as db:

			if "means" in db.tables:
				raise ValueError("Means for database {0} already computed!".format(dbname))

			#Cycle over models
			for n,m in enumerate(models):
				
				#Query the database
				logdriver.info("Processing model {0} ({1} of {2})".format(m,n,len(models)))
				sql_query = "SELECT * FROM features WHERE model={0}".format(m)
				logdriver.info("SQL: {0}".format(sql_query))
				realizations = db.query(sql_query)

				#Compute the means by redshift
				means = realizations.groupby(getattr(settings,feature).redshift_labels).mean(0).reset_index()

				#Append model row as an entry into the "models" table
				models_row = pd.Series(means[parameter_names].mean(),columns=parameter_names)
				models_row["model"] = m
				db.insert(pd.DataFrame(models_row).T,"models") 

				#Pop these useless columns
				for c in ["realization","sub_catalog"] + parameter_names:
					means.pop(c)

				#Insert means into "means" table
				db.insert(means,"means")


if __name__=="__main__":
	main()


