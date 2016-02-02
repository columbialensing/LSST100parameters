#!/usr/bin/env python

import sys,logging
sys.modules["mpi4py"] = None

import argparse,json

from library.featureDB import LSSTSimulationBatch
from library.driver_mpi import cosmo_constraints
from library.defaults import settings

#Command line options
parser = argparse.ArgumentParser()
parser.add_argument("-e","--environment",dest="environment",action="store",default="environment.ini",help="INI file with the batch location")
parser.add_argument("-f","--features",dest="features",action="store",default=None,help="JSON file containing the specifications of the features to combine, in None, these are read from stdin")
parser.add_argument("-v","--verbose",dest="verbose",action="store_true",default=False,help="Turn on verbosity")

#Create a list of single redshift features out of a tomographic one
def split_redshifts(specs,redshift_index=range(5)):

	#List of splittings
	splitted_specs = list()

	#Cycle over features and redshift bin
	for zi in redshift_index:
		
		#Deep copy of the original dictionary
		specs_redshift = dict()
		for key in ["dbname","table_name","feature_label_root","feature_label_format","features","realizations_for_covariance","realizations_for_data","pca_components"]:
			specs_redshift[key] = specs[key]

		#Append redshift label to name
		specs_redshift["feature_label_root"] += "_z{0}".format(zi)
		
		#Update features with redshift filter
		for feature in specs["features"]:
			specs_redshift[feature] = dict((("feature_filter",specs[feature]["feature_filter"]),("realization_filter",specs[feature]["realization_filter"])))
			specs_redshift[feature]["redshift_filter"] = " AND ".join(["{0}={1}".format(l,zi) for l in getattr(settings,feature).redshift_labels])
	
		#Append to list
		splitted_specs.append(specs_redshift)

	#Return to user
	return splitted_specs

############################################################################################################################################################################

#Main
def main():

	#Parse arguments
	cmd_args = parser.parse_args()

	#Verbosity level
	if cmd_args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.INFO)

	#Init batch
	batch = LSSTSimulationBatch.current(cmd_args.environment)

	#Read json specs and sanitize input
	if cmd_args.features is not None:
		with open(cmd_args.features,"r") as fp:
			specs = json.load(fp)
	else:
		specs = json.loads(sys.stdin.read())

	if type(specs)==dict:
		specs = [specs]

	#Cycle over specifications
	for s in specs:

		#Sanitize None
		for feature in s["features"]:
			for l in ["feature_filter","redshift_filter","realization_filter"]:
				if s[feature][l]=="None":
					s[feature][l] = None
		
		#Execute
		cosmo_constraints(batch,s,settings)


if __name__=="__main__":
	main()

