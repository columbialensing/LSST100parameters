#!/usr/bin/env python-mpi

import library.driver_mpi as driver
from lenstools.statistics.ensemble import Ensemble

import numpy as np
import pandas as pd

#Measure the cross spectrum of a list of convergence maps
def cross_power(maps,ell_edges,indices):

	"""

	:param maps: list of convergence maps (one for each redshift bin)
	:type maps: list.
		
	:param ell_edges: multipoles
	:type ell_edges: array

	:param indices: pairs of indices; each pair corresponds to a pair of redshift bins to cross correlate
	:type indices: list of tuples

	:returns: Ensemble

	"""

	#Allocate memory
	cross_power_array = np.zeros((len(indices),len(ell_edges)-1))
		
	#Measure the auto and cross power spectrum
	for n,(i,j) in enumerate(indices):
		ell,cross_power_array[n] = maps[i].cross(maps[j],statistic="power_spectrum",l_edges=ell_edges)

	#Build the Ensemble
	columns = [ "l{0}".format(n) for n in range(1,len(ell)+1) ]
	ensemble = Ensemble(cross_power_array,columns=columns)

	#Add the indices labels
	ensemble["b1"],ensemble["b2"] = zip(*indices)

	#Return
	return ensemble

#Redshift bin index pairs and multipoles
measurer_kwargs = {
	
"measurer" : cross_power,
"indices" : zip(*np.triu_indices(5)),
"ell_edges" : np.array([  100.,   238.,   376.,   514.,   652.,   790.,   928.,  1066., 
	1204.,  1342.,  1480.,  1618.,  1756.,  1894.,  2032.,  2170.,
	2308.,  2446.,  2584.,  2722.,  2860.,  2998.,  3136.,  3274.,
	3412.,  3550.,  3688.,  3826.,  3964.,  4102.,  4240.,  4378.,
	4516.,  4654.,  4792.,  4930.,  5068.,  5206.,  5344.,  5482.,
	5620.,  5758.,  5896.,  6034.,  6172.,  6310.,  6448.,  6586.,
	6724.,  6862.,  7000.])
	
}

default_db_name = "cross_spectra"

###############################################################################################

if __name__=="__main__":
	driver.measure_main(measurer_kwargs,default_db_name)