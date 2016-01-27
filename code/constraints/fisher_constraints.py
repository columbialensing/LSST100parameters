#!/usr/bin/env python

import sys
sys.modules["mpi4py"] = None

import argparse,json

from lenstools import SimulationBatch
from library.driver_mpi import cosmo_constraints

#Command line options
parser = argparse.ArgumentParser()
parser.add_argument("-e","--environment",dest="environment",action="store",default="environment.ini",help="INI file with the batch location")
parser.add_argument("-f","--features",dest="features",action="store",default="combine_default.json",help="JSON file containing the specifications of the features to combine")

#Main
def main():

	#Parse arguments
	cmd_args = parser.parse_args()

	#Init batch
	batch = SimulationBatch.current(cmd_args.environment)

	#Read json specs and sanitize input
	with open(cmd_args.features,"r") as fp:
		specs = json.load(fp)

	for feature in specs["features"]:
		for l in ["feature_filter","redshift_filter","realization_filter"]:
			if specs[feature][l]=="None":
				specs[feature][l] = None
		
	#Execute
	cosmo_constraints(batch,specs)


if __name__=="__main__":
	main()

