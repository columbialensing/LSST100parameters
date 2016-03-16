#!/usr/bin/env python-mpi

import library.driver_mpi as driver
from lenstools.statistics.ensemble import Ensemble

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

#Redshift bin index pairs and multipoles
measurer_kwargs = {
	
"measurer" : peaks,
"indices" : range(5),
"kappa_edges" : pd.read_pickle("/global/homes/a/apetri/LSST100Parameters/data/edges.pkl")["kappa_edges_sigma"].values
	
}

default_db_name = "peaks"

###############################################################################################

if __name__=="__main__":
	driver.measure_main(measurer_kwargs,default_db_name)