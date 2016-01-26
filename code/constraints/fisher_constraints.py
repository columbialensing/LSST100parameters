#!/usr/bin/env python

import sys
sys.modules["mpi4py"] = None

import json

from lenstools import SimulationBatch
from library.driver_mpi import cosmo_constraints

def main():

	#Init batch
	batch = SimulationBatch.current()

	#Read json specs and sanitize input
	with open("combine_default.json","r") as fp:
		specs = json.load(fp)

	for feature in specs["features"]:
		for l in ["feature_filter","redshift_filter","realization_filter"]:
			if specs[feature][l]=="None":
				specs[feature][l] = None
		
	#Execute
	cosmo_constraints(batch,specs)


if __name__=="__main__":
	main()

