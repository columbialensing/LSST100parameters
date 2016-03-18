#!/usr/bin/env python

import sys,argparse
sys.modules["mpi4py"] = None

import numpy as np
import matplotlib.pyplot as plt

from library.featureDB import FisherDatabase

#Options
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",dest="type",default="png",help="format of the figure to save")
parser.add_argument("fig",nargs="*")

#Plot labels
par2label = {

"Om" : r"$\Omega_m$" ,
"w" : r"$w$" ,
"sigma8" : r"$\sigma_8$"

}

#Fiducial value
par2value = {

"Om" : 0.26 ,
"w" : -1. ,
"sigma8" : 0.8

}

###################################################################################################

def photoz_bias(cmd_args,db_name="data/fisher/constraints_photoz.sqlite",parameters=["Om","w"],feature_label="power_spectrum_pca",fontsize=22):
	
	#Init figure
	fig,ax = plt.subplots()

	############
	#No photo-z#
	############

	with FisherDatabase(db_name) as db:
		pfit = db.query_parameter_fit(feature_label,table_name="shape_noise",parameters=parameters)
		p1,p2 = [ pfit[parameters[n]+"_fit"].values for n in [0,1] ]
		ax.scatter(p1,p2,s=pfit["bins"],color="black",label="No photo-z")

	##############
	#With photo-z#
	##############

	with FisherDatabase(db_name) as db:
		pfit = db.query_parameter_fit(feature_label,table_name="shape_noise_photoz",parameters=parameters)
		p1,p2 = [ pfit[parameters[n]+"_fit"].values for n in [0,1] ]
		ax.scatter(p1,p2,s=pfit["bins"],color="red",label="With photo-z")

	#Get axes bounds
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	#Show the fiducial value
	ax.plot(np.ones(100)*par2value[parameters[0]],np.linspace(*(ylim+(100,))),linestyle="-",color="green")
	ax.plot(np.linspace(*(xlim+(100,))),np.ones(100)*par2value[parameters[1]],linestyle="-",color="green")

	#Set the axes bounds
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	#Legends
	ax.set_xlabel(par2label[parameters[0]],fontsize=fontsize)
	ax.set_ylabel(par2label[parameters[1]],fontsize=fontsize)
	ax.legend()

	#Save figure
	fig.savefig("photoz_bias_{0}.".format(feature_label)+cmd_args.type)


def photoz_bias_power_spectrum(cmd_args):
	photoz_bias(cmd_args,feature_label="power_spectrum_pca")

def photoz_bias_peaks(cmd_args):
	photoz_bias(cmd_args,feature_label="peaks_pca")

def photoz_bias_moments(cmd_args):
	photoz_bias(cmd_args,feature_label="moments_pca")

###########################################################################################################################################

#Method dictionary
method = dict()
method["1a"] = photoz_bias_power_spectrum
method["1b"] = photoz_bias_peaks
method["1c"] = photoz_bias_moments

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()