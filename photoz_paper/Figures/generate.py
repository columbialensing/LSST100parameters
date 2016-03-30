#!/usr/bin/env python

import sys,os,argparse
sys.modules["mpi4py"] = None

import numpy as np
import matplotlib.pyplot as plt

from lenstools.catalog.shear import Catalog
from lenstools.statistics.constraints import FisherAnalysis
from lenstools.simulations.design import Design

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

def design(cmd_args,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots(1,2,figsize=(16,8))

	#Read in the simulation design
	design = Design.read("md/design.pkl")
	parameters = [("Om","w"),("Om","sigma8")]

	#Plot the points
	for n,(p1,p2) in enumerate(parameters):
		ax[n].scatter(design[p1],design[p2],marker="x")
		ax[n].scatter(par2value[p1],par2value[p2],marker="x",color="red",s=50)
		ax[n].set_xlabel(par2label[p1],fontsize=fontsize)
		ax[n].set_ylabel(par2label[p2],fontsize=fontsize)

	#Save figure
	fig.savefig("design."+cmd_args.type)

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
	ax.set_ylabel(r"$N_g(z)$",fontsize=fontsize)
	ax.legend()

	#Save figure
	fig.savefig("galdistr."+cmd_args.type)

###################################################################################################
###################################################################################################

def scibook_photo(cmd_args,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()

	#Read in bias and sigma
	z,b = np.loadtxt("data/photoz/SciBook_bias_gold.out",unpack=True)
	z,s = np.loadtxt("data/photoz/SciBook_sigma_gold.out",unpack=True)

	#Plot
	ax.plot(z,b,label=r"$b_{\rm ph}(z_s)$ ${\rm optimistic}$",color="black")
	ax.plot(z,s,label=r"$\sigma_{\rm ph}(z_s)$ ${\rm optimistic}$",color="red")

	#Read in bias and sigma
	z,b = np.loadtxt("data/photoz/weightcombo_bias_gold.out",unpack=True)
	z,s = np.loadtxt("data/photoz/weightcombo_sigma_gold.out",unpack=True)

	#Plot
	ax.plot(z,b,label=r"$b_{\rm ph}(z_s)$ ${\rm pessimistic}$",color="black",linestyle="--")
	ax.plot(z,s,label=r"$\sigma_{\rm ph}(z_s)$ ${\rm pessimistic}$",color="red",linestyle="--")

	#Labels
	ax.set_xlabel(r"$z_s$",fontsize=fontsize)
	ax.legend(loc="upper left")

	#Save figure
	fig.savefig("scibook."+cmd_args.type)


###################################################################################################
###################################################################################################

def pca_components(cmd_args,db_name="data/fisher/constraints_combine.sqlite",feature_label="power_spectrum_pca",parameter="w",fontsize=22):

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

features_to_show = {

"ps" : {"name":"power_spectrum_pca","table_name" : "pcov_noise", "pca_components" : 30, "color" : "red", "label" : r"$P^{\kappa\kappa}(30)$","linestyle" : "-"},
"ps70" : {"name":"power_spectrum_pca","table_name" : "pcov_noise", "pca_components" : 70, "color" : "red", "label" : r"$P^{\kappa\kappa}(70)$","linestyle" : "--"},

"mu" : {"name":"moments_pca","table_name" : "pcov_noise", "pca_components" : 30, "color" : "blue", "label" : r"$\mathbf{\mu}(30)$","linestyle" : "-"},
"mu40" : {"name":"moments_pca","table_name" : "pcov_noise", "pca_components" : 40, "color" : "blue", "label" : r"$\mathbf{\mu}(40)$","linestyle" : "--"},

"pk" : {"name":"peaks_pca", "table_name": "pcov_noise", "pca_components" : 40, "color" : "green", "label" : r"$n_{\rm pk}(40)$","linestyle" : "-"},
"pk70" : {"name":"peaks_pca", "table_name": "pcov_noise", "pca_components" : 70, "color" : "green", "label" : r"$n_{\rm pk}(70)$","linestyle" : "--"},

"ps+pk" : {"name" : "power_spectrum+peaks" , "table_name" : "pcov_noise_combine", "pca_components" : 30+40, "color" : "orange", "label" : r"$P^{\kappa\kappa}(30)+n_{\rm pk}(40)$","linestyle" : "-"},
"ps+mu" : {"name" : "power_spectrum+moments" , "table_name" : "pcov_noise_combine", "pca_components" : 30+30, "color" : "purple", "label" : r"$P^{\kappa\kappa}(30)+\mathbf{\mu}(30)$","linestyle" : "-"},
"ps+pk+mu" : {"name" : "power_spectrum+peaks+moments" , "table_name" : "pcov_noise_combine", "pca_components" : 30+30+40, "color" : "black", "label" : r"$P^{\kappa\kappa}(30)+n_{\rm pk}(40)+\mathbf{\mu}(30)$","linestyle" : "-"},

"order" : ["ps","ps70","pk","pk70","mu","mu40","ps+pk","ps+mu","ps+pk+mu"]

}

def parameter_constraints(cmd_args,db_name="data/fisher/constraints_combine.sqlite",features_to_show=features_to_show,parameters=["Om","w"],xlim=(0.25,0.27),ylim=(-1.06,-0.94),fontsize=22):

	#Init figure
	fig,ax = plt.subplots()
	ellipses = list()
	labels = list()

	#Plot the features 
	with FisherDatabase(db_name) as db:
		for f in features_to_show["order"]:

			#Query parameter covariance
			pcov = db.query_parameter_covariance(features_to_show[f]["name"],nbins=features_to_show[f]["pca_components"],table_name=features_to_show[f]["table_name"],parameters=parameters)

			#Show the ellipse
			center = (par2value[parameters[0]],par2value[parameters[1]])
			ellipse = FisherAnalysis.ellipse(center=center,covariance=pcov.values,p_value=0.677,fill=False,edgecolor=features_to_show[f]["color"],linestyle=features_to_show[f]["linestyle"])
			ax.add_artist(ellipse)

			#Labels
			ellipses.append(ellipse)
			labels.append(features_to_show[f]["label"])

	#Axes bounds
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)

	#Axes labels and legend
	ax.set_xlabel(par2label[parameters[0]],fontsize=fontsize)
	ax.set_ylabel(par2label[parameters[1]],fontsize=fontsize)
	ax.legend(ellipses,labels,bbox_to_anchor=(0.,1.02,1.,.102),loc=3,ncol=3,mode="expand",borderaxespad=0.)

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

			#Draw an error ellipse around the mean bias
			center = ((p1-p1f).mean(),(p2-p2f).mean())
			pcov = np.cov([p1-p1f,p2-p2f]) 
			ax.add_artist(FisherAnalysis.ellipse(center,pcov,p_value=0.677,fill=False,edgecolor=colors[nf]))

	#Get axes bounds
	xlim = np.abs(np.array(ax.get_xlim())).max()
	ylim = np.abs(np.array(ax.get_ylim())).max()

	#Show the fiducial value
	ax.plot(np.zeros(100),np.linspace(-ylim,ylim,100),linestyle="-",color="green")
	ax.plot(np.linspace(-xlim,xlim,100),np.zeros(100),linestyle="-",color="green")

	#Set the axes bounds
	ax.set_xlim(-xlim,xlim)
	ax.set_ylim(-ylim,ylim)

	#Legends
	ax.set_xlabel(r"$\delta$" + par2label[parameters[0]],fontsize=fontsize)
	ax.set_ylabel(r"$\delta$" + par2label[parameters[1]],fontsize=fontsize)
	ax.legend(loc="upper left")

	#Save figure
	fig.savefig("photoz_bias."+cmd_args.type)

###################################################################################################
###################################################################################################
###################################################################################################

#Method dictionary
method = dict()

method["1"] = design
method["2"] = galdistr
method["3"] = scibook_photo

method["4a"] = pca_components_power_spectrum
method["4b"] = pca_components_peaks
method["4c"] = pca_components_moments

method["5"] = parameter_constraints
method["6"] = photoz_bias


#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()