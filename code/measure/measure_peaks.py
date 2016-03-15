#!/usr/bin/env python-mpi
import argparse,logging

import library.driver_mpi as driver
from library.featureDB import LSSTSimulationBatch
from lenstools.statistics.ensemble import Ensemble
from lenstools.pipeline.settings import EnvironmentSettings

import numpy as np
import pandas as pd

#Measure the peak counts
def peaks(maps,kappa_edges,indices):

	"""

	:param maps: list of convergence maps (one for each redshift bin)
	:type maps: list.
		
	:param kappa_edges: convergence value bin edges
	:type kappa_edges: array

	:returns: Ensemble

	"""

	#Allocate memory
	peaks_array = np.zeros((len(indices),len(kappa_edges)-1))
		
	#Measure the peak counts
	for n,i in enumerate(indices):
		kappa,peaks_array[n] = maps[i].peakCount(kappa_edges,norm=True)

	#Build the Ensemble
	columns = [ "k{0}".format(n) for n in range(1,len(kappa)+1) ]
	ensemble = Ensemble(peaks_array,columns=columns)

	#Add the indices labels
	ensemble["b1"] = indices

	#Return
	return ensemble

###############################################################################################

if __name__=="__main__":

	########################################################################################################################

	#Redshift bin index pairs and multipoles
	measurer_kwargs = {
	
	"measurer" : peaks,
	"indices" : range(5),
	"kappa_edges" : pd.read_pickle("/global/homes/a/apetri/LSST100Parameters/data/edges.pkl")["kappa_edges_sigma"].values
	
	}

	#############################################################
	###########The part below should be standard#################
	#############################################################

	#Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--environment",dest="environment",help="Environment options file")
	parser.add_argument("-c","--config",dest="config",help="Configuration file")

	parser.add_argument("-n","--noise",dest="noise",action="store_true",default=False,help="Add shape noise")
	parser.add_argument("-pb","--photoz_bias",dest="photoz_bias",action="store",default=None,help="Read photoz biases from this file")
	parser.add_argument("-ps","--photoz_sigma",dest="photoz_sigma",action="store",default=None,help="Read photoz sigmas from this file")
	
	parser.add_argument("-d","--database",dest="database",default="peaks",help="Database name to populate")
	parser.add_argument("id",nargs="*")
	cmd_args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)

	#Handle on the current batch
	batch = LSSTSimulationBatch(EnvironmentSettings.read(cmd_args.environment))

	#Output database name
	database_name = cmd_args.database
	
	if cmd_args.noise:
		database_name += "_noise" 
	
	database_name += ".sqlite"

	driver_kwargs = {

	"db_name" : database_name,
	"add_shape_noise" : cmd_args.noise,
	"photoz_bias" : cmd_args.photoz_bias,
	"photoz_sigma" : cmd_args.photoz_sigma,
	"pool" : None

	}

	#Merge keyword arguments dictionaries
	driver_measurer_kwargs = dict(driver_kwargs,**measurer_kwargs)

	#Execute
	for model_id in cmd_args.id:
		
		#Parse cosmo_id and model number
		cosmo_id,n = model_id.split("|")
		
		#Check conditions for table names
		if cosmo_id==batch.fiducial_cosmo_id:
			
			if (cmd_args.photoz_bias is not None) or (cmd_args.photoz_sigma is not None):
				catalog2table = {"Shear":"features_fiducial_photoz","ShearEmuIC":"features_fiducial_EmuIC_photoz"}
			else:
				catalog2table = {"Shear":"features_fiducial","ShearEmuIC":"features_fiducial_EmuIC"}

		else:

			if (cmd_args.photoz_bias is not None) or (cmd_args.photoz_sigma is not None):
				catalog2table = {"Shear":"features_photoz"}
			else:
				catalog2table = {"Shear":"features"}

		#Execution
		driver.measure(batch=batch,cosmo_id=cosmo_id,model_n=int(n),catalog2table=catalog2table,**driver_measurer_kwargs)