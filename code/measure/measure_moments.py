#!/usr/bin/env python-mpi

import library.driver_mpi as driver
from lenstools.statistics.ensemble import Ensemble

import numpy as np
import pandas as pd

#Measure the moments of a list of convergence maps
def moments(maps,indices):

	"""

	:param maps: list of convergence maps (one for each redshift bin)
	:type maps: list.

	:returns: Ensemble

	"""

	#Allocate memory
	moments_array = np.zeros((len(indices),9))
		
	#Measure the moments
	for n,i in enumerate(indices):
		moments_array[n] = maps[i].moments(connected=True)

	#Build the Ensemble
	columns = [ "sigma0","sigma1","S0","S1","S2","K0","K1","K2","K3" ]
	ensemble = Ensemble(moments_array,columns=columns)

	#Add the indices labels
	ensemble["b1"] = indices

	#Return
	return ensemble

measurer_kwargs = {
	
"measurer" : moments,
"indices" : range(5)
	
}

default_db_name = "moments"

###############################################################################################

if __name__=="__main__":
	driver.measure_main(measurer_kwargs,default_db_name)
