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
"kappa_edges" : np.array([ -2. ,  -1.7,  -1.4,  -1.1,  -0.8,  -0.5,  -0.2,   0.1,   
	0.4, 0.7,   1. ,   1.3,   1.6,   1.9,   2.2,   2.5,   2.8,   3.1, 3.4,   3.7,   
	4. ,   4.3,   4.6,   4.9,   5.2,   5.5,   5.8, 6.1,   6.4,   6.7,   7. , 7.3,   
	7.6,   7.9,   8.2,   8.5, 8.8,   9.1,   9.4,   9.7,  10. ,  10.3,  10.6,  10.9,  
	11.2, 11.5,  11.8,  12.1,  12.4,  12.7,  13. ])
	
}

default_db_name = "peaks"

###############################################################################################

if __name__=="__main__":
	driver.measure_main(measurer_kwargs,default_db_name)