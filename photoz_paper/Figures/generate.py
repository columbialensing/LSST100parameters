#!/usr/bin/env python

import sys,os,argparse
sys.modules["mpi4py"] = None

import numpy as np
import matplotlib.pyplot as plt

from lenstools.catalog.shear import Catalog

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
###################################################################################################

def galdistr(cmd_args,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()

	#Read in galaxy redshifts
	colors = ["blue","green","red","purple","yellow"]
	position_files = [os.path.join("data","positions_bin{0}.fits".format(n)) for n in range(1,6)]
	for n,f in enumerate(position_files):
		z = Catalog.read(f)["z"]

		#Make the histogram
		ng,zb = np.histogram(z,bins=np.arange(z.min(),z.max(),0.02))
		ax.plot(0.5*(zb[1:]+zb[:-1]),ng,color=colors[n],label=r"$z\in[{0:.2f},{1:.2f}]$".format(z.min(),z.max()))
		ax.fill_between(0.5*(zb[1:]+zb[:-1]),np.zeros_like(ng),ng,color=colors[n],alpha=0.3)

	#Labels
	ax.set_xlabel(r"$z$",fontsize=fontsize)
	ax.set_ylabel(r"$N_g$",fontsize=fontsize)
	ax.legend()

	#Save figure
	fig.savefig("galdistr."+cmd_args.type)

###################################################################################################
###################################################################################################

def pca_components(cmd_args,db_name="data/fisher/constraints.sqlite",feature_label="power_spectrum_pca",parameter="w",fontsize=22):

	#Use automatic plot routine
	with FisherDatabase(db_name) as db:
		fig,axes = db.plot_by_feature([feature_label],table_name="pcov_noise",parameter="w")

	#Labels
	axes[0].set_ylabel(r"$\Delta$"+par2label[parameter],fontsize=fontsize)

	#Save figure
	fig.savefig("{0}_{1}.".format(parameter,feature_label)+cmd_args.type)

def pca_components_power_spectrum(cmd_args):
	pca_components(cmd_args,feature_label="power_spectrum_pca")

def pca_components_peaks(cmd_args):
	pca_components(cmd_args,feature_label="peaks_pca")

def pca_components_moments(cmd_args):
	pca_components(cmd_args,feature_label="moments_pca")

###################################################################################################
###################################################################################################

features_to_show = 
{
	"power_spectrum_pca" : {"table_name" : "pcov_noise", "pca_components" : 60, "color" : "red"},
	"moments_pca" : {"table_name" : "pcov_noise", "pca_components" : 60, "color" : "blue"},
}

def parameter_constraints(cmd_args,db_name="data/fisher/constraints.sqlite",features_to_show=features_to_show,parameters=["Om","w"],fontsize=22):

	#Init figure
	fig,ax = plt.subplots()

	#Save figure
	fig.savefig("constraints."+cmd_args.type)

###################################################################################################
###################################################################################################

def photoz_bias(cmd_args,db_name="data/fisher/constraints_photoz.sqlite",parameters=["Om","w"],feature_labels=["power_spectrum_pca","peaks_pca","moments_pca"],colors=["black","red","blue"],fontsize=22):
	
	#Init figure
	fig,ax = plt.subplots()

	#Cycle over features
	for nf,f in enumerate(feature_labels):

		############
		#No photo-z#
		############

		with FisherDatabase(db_name) as db:
			pfit = db.query_parameter_fit(f,table_name="shape_noise_mocks",parameters=parameters)
			p1f,p2f = [ pfit[parameters[n]+"_fit"].values for n in [0,1] ]

		##########################
		#With photo-z: optimistic#
		##########################

		with FisherDatabase(db_name) as db:
			pfit = db.query_parameter_fit(f,table_name="shape_noise_mocks_photoz",parameters=parameters)
			p1,p2 = [ pfit[parameters[n]+"_fit"].values for n in [0,1] ]
			ax.scatter(p1-p1f,p2-p2f,color=colors[nf],marker=".",label=(nf==0 or None) and "With photo-z (optimistic)")

		###########################
		#With photo-z: pessimistic#
		###########################

		with FisherDatabase(db_name) as db:
			pfit = db.query_parameter_fit(f,table_name="shape_noise_mocks_photoz_pessimistic",parameters=parameters)
			p1,p2 = [ pfit[parameters[n]+"_fit"].values for n in [0,1] ]
			ax.scatter(p1-p1f,p2-p2f,color=colors[nf],marker="x",label=(nf==0 or None) and "With photo-z (pessimistic)")
			ax.scatter((p1-p1f).mean(),(p2-p2f).mean(),color=colors[nf],marker="s",s=60)

	#Get axes bounds
	xlim = np.abs(np.array(ax.get_xlim())).max()
	ylim = np.abs(np.array(ax.get_ylim())).max()

	#Show the fiducial value
	ax.plot(np.ones(100)*par2value[parameters[0]],np.linspace(-ylim,ylim,100),linestyle="-",color="green")
	ax.plot(np.linspace(-xlim,xlim,100),np.ones(100)*par2value[parameters[1]],linestyle="-",color="green")

	#Set the axes bounds
	ax.set_xlim(-xlim,xlim)
	ax.set_ylim(-ylim,ylim)

	#Legends
	ax.set_xlabel(r"${\rm bias}$"+ "$($" + par2label[parameters[0]] + "$)$",fontsize=fontsize)
	ax.set_ylabel(r"${\rm bias}$"+ "$($" + par2label[parameters[1]] + "$)$",fontsize=fontsize)
	ax.legend(loc="upper left")

	#Save figure
	fig.savefig("photoz_bias."+cmd_args.type)

###################################################################################################
###################################################################################################
###################################################################################################

#Method dictionary
method = dict()
method["1"] = galdistr

method["2a"] = pca_components_power_spectrum
method["2b"] = pca_components_peaks
method["2c"] = pca_components_moments

method["4"] = photoz_bias


#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()