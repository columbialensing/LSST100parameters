from __future__ import division
import cPickle as pkl
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

def generate_gaussian_photoz_errors(z,intervals=None,bias=None,sigma=None,seed=None):

	#Set random seed 
	if seed is not None:
		np.random.seed(seed)

	#Vector for photoz errors
	photoz_errors = np.zeros_like(z)

	#Generate errors with bias + sigma*N(0,1)
	if bias is not None:
		
		if type(bias) in [str,unicode]:
			
			if bias.endswith(".pkl") or bias.endswith(".p"):
				
				#Deserialize bias calculation function and apply it to z
				with open(bias,"r") as fp:
					photoz_errors += pkl.load(fp)(z)

			else:
				intervals,bias_values = parse_photoz_txt(bias)
				photoz_errors += step(z,intervals,bias_values)
		
		elif type(bias)==float:
			photoz_errors += bias*(1+z)
		
		else:
			intervals,bias_values = intervals,bias
			photoz_errors += step(z,intervals,bias_values)
		

	if sigma is not None:

		if type(sigma) in [str,unicode]:
			
			if sigma.endswith(".pkl") or sigma.endswith(".p"):
				
				#Deserialize sigma calculation function and apply it to z
				with open(sigma,"r") as fp:
					photoz_errors += np.random.randn(len(z))*pkl.load(fp)(z)

			else:
				intervals,sigma_values = parse_photoz_txt(sigma)
				photoz_errors += np.random.randn(len(z))*step(z,intervals,sigma_values)
		
		elif type(sigma)==float:
			photoz_errors += np.random.randn(len(z))*sigma*(1+z)

		else:
			intervals,sigma_values = intervals,sigma
			photoz_errors += np.random.randn(len(z))*step(z,intervals,sigma_values)

	
	#Retrurn to user
	return photoz_errors