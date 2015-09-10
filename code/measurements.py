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

class FeatureDatabase(object):

	###############################################
	#Create convergence maps out of shear catalogs#
	###############################################
	
	@staticmethod
	def make_maps(shear_files,position_files,npixel=512,smooth=0.5*u.arcmin,fov=3.5*u.deg):

		"""
		:param shear_files: list of files that contain the shear catalogs
		:type shear_files: list.

		:param position_files: list of files that contain positions and redshift of the galaxies in the shear files
		:type position_files: list.

		:returns: list of ConvergenceMap

		"""

		#Safety checks
		assert len(shear_files)==len(position_files)

		#Read in all the shear catalogs, and convert to convergence maps with E/B mode decomposition
		convergence_maps = list()
		for n,shear_file in enumerate(shear_files):
			full_catalog = tbl.hstack((ShearCatalog.read(position_files[n]),ShearCatalog.read(shear_file)))
			convergence_maps.append(full_catalog.toMap(map_size=fov,npixel=npixel,smooth=smooth).convergence())

		#Return
		return convergence_maps
	

	##########################################################
	#Measure the cross spectrum of a list of convergence maps#
	##########################################################

	@staticmethod
	def cross_power(maps,l_edges,indices,**kwargs):

		"""

		:param maps: list of convergence maps (one for each redshift bin)
		:type maps: list.

		:param l_edges: multipoles
		:type l_edges: array

		:param indices: pairs of indices; each pair corresponds to a pair of redshift bins to cross correlate
		:type indices: list of tuples

		:returns: Ensemble

		"""

		#Allocate memory
		cross_power_array = np.zeros((len(indices),len(l_edges)-1))
		
		#Measure the auto and cross power spectrum
		for n,(i,j) in enumerate(indices):
			ell,cross_power_array[n] = maps[i].cross(maps[j],statistic="power_spectrum",l_edges=l_edges)

		#Build the Ensemble
		columns = ["b1","b2"] + [ "l{0}".format(n) for n in range(1,len(ell)+1) ]
		ensemble = Ensemble(np.hstack((np.array(indices),cp)),columns=columns)

		#Return
		return ensemble


