from __future__ import division
import numpy as np
from lenstools.utils.algorithms import step

##############################
#Parse photoz bias and sigmas#
##############################

def parse_photoz_txt(fname):

	#Parse ASCII file
	z,v = np.loadtxt(fname,unpack=True)

	#Construct the intervals
	left = z[:-1]
	right = z[1:]

	#Return tuple (intervals,values)
	return zip(left,right),v[1:]

#################################
#Generate Gaussian photoz errors#
#################################

def generate_gaussian_photoz_errors(z,intervals,bias,sigma,seed=None):

	#Set random seed 
	if seed is not None:
		np.random.seed(seed)

	#Generate errors with bias + sigma*N(0,1)
	return step(z,intervals,bias) + np.random.randn(len(z))*step(z,intervals,sigma)