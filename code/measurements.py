############################################
#Measurements of features on shear catalogs#
############################################

from __future__ import division,print_function,with_statement
import sys,os

from lenstools.catalog import ShearCatalog
from lenstools.statistics.ensemble import Ensemble

import numpy as np
import astropy.units as u
import astropy.table as tbl

#Measure the cross power spectrum between bins in each realization
def cross_power(shear_files,position_files,l_edges,indices,npixel=512,smooth=0.5*u.arcmin,fov=3.5*u.deg):

	"""
	:param shear_files: list of files that contain the shear catalogs
	:type shear_files: list.

	:param position_files: list of files that contain positions and redshift of the galaxies in the shear files
	:type position_files: list.

	:param l_edges: multipoles
	:type l_edges: array

	:param indices: pairs of indices; each pair corresponds to a pair of redshift bins to cross correlate
	:type indices: list of tuples

	:returns: ell,cross_power

	"""

	#Safety checks
	assert len(shear_files)==len(position_files)

	#Read in all the shear catalogs, and convert to convergence maps with E/B mode decomposition
	convergence_maps = list()
	for n,shear_file in enumerate(shear_files):
		full_catalog = tbl.hstack((ShearCatalog.read(position_files[n]),ShearCatalog.read(shear_file)))
		convergence_maps.append(full_catalog.toMap(map_size=fov,npixel=npixel,smooth=smooth).convergence())

	#################################################################################################################
	#Now that the maps have been read in, we can compute the auto and cross power spectra at the selected multipoles#
	#################################################################################################################

	#Allocate memory
	cross_power_array = np.zeros((len(indices),len(l_edges)-1))

	#Measure the auto and cross power spectrum
	for n,(i,j) in enumerate(indices):
		ell,cross_power_array[n] = convergence_maps[i].cross(convergence_maps[j],statistic="power_spectrum",l_edges=l_edges)

	#Return the result
	return ell,cross_power_array

#Measure the cross spectrum of a particular realization in a simulation model
def cross_power_ensemble(realization,model,l_edges,indices,**kwargs):

	"""

	:param realization: realization to measure
	:type realization: int.

	:param model: cosmological model to consider
	:type model: SimulationModel

	:param l_edges: multipoles
	:type l_edges: array

	:param indices: pairs of indices; each pair corresponds to a pair of redshift bins to cross correlate
	:type indices: list of tuples

	:returns: Ensemble

	"""

	#Build filenames
	shear_files = [ model.getCollection("512b260").getCatalog("Shear").path("WLshear_positions_bin{0}_{1:04d}r.fits".format(n,realization)) for n in range(1,6) ]
	position_files = [ "../data/positions_bin{0}.fits".format(n) for n in range(1,6) ]

	#Measure
	ell,cp = cross_power(shear_files,position_files,l_edges,indices,**kwargs)

	#Build the Ensemble
	columns = ["b1","b2","r"] + [ "l{0}".format(n) for n in range(1,len(ell)+1) ]
	ensemble = Ensemble(np.hstack((np.array(indices),np.ones(len(indices),dtype=np.int)[:,None]*realization,cp)),columns=columns)

	#Return
	return ensemble


