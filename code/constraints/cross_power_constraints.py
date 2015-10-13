#!/usr/bin/env python-mpi

import sys
import logging
import numpy as np

from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.constraints import Emulator
from lenstools.statistics.database import chi2database

from library.featureDB import FeatureDatabase
from library.algorithms import Manifold

logging.basicConfig(level=logging.INFO)

#Score parameters
multipole_names = ["l{0}".format(n) for n in range(1,51)]
score_parameters = Ensemble.meshgrid({"Om":np.linspace(0.256,0.2635,100),"w":np.linspace(-1.08,-0.92,100)})
score_parameters["sigma8"] = 0.8

#Fiducial parameters
p_fiducial = Ensemble(np.array([0.26,-1.0,0.8])[None],columns=["Om","w","sigma8"])
p_fiducial.add_name("parameters")

#Number of noise directions
n_noise = 10

#Specifications
specs = dict()

#Open the fiducial cosmology database for data and covariances
with FeatureDatabase("../data/features/cross_spectra_fiducial.sqlite") as db:

	#Name the feature
	feature_name = "cross_power"

	#Read the emulator
	emulator = Emulator.read("../data/emulators/emulator_power_cross_all.pkl")

	#Predict the feature at the fiducial cosmology
	fiducial_feature = emulator.predict(np.array([0.26,-1.0,0.8]))

	#Extract the directions natural to Om and w
	manifold = Manifold(emulator)
	tangents = manifold.draw_tangents()
	v = (tangents["Om"].iloc[1],tangents["w"].iloc[1])

	#Refeaturize the emulator projecting on the principal components
	emulator_pca = emulator.refeaturize(lambda f:manifold.pca.transform(f),method="apply_whole")

	#Get the data and project on the principal components
	lab,pow = db.query("SELECT realization,b1,b2,{0} FROM features_fiducial".format(",".join(multipole_names))).suppress_indices(by=["realization"],suppress=["b1","b2"],columns=multipole_names)
	pow.pop("realization")
	pow.add_name(feature_name)
	pow = manifold.pca.transform(pow)

	#Extract a bunch of directions proper to the noise (cosmic variance)
	fiducial_feature_pca = manifold.pca.transform(fiducial_feature)
	dn_pca = pow.tail(n_noise).apply(lambda r:r-fiducial_feature_pca,axis=1).apply(lambda r:r/np.sqrt(r.dot(r)),axis=1)

	#Compute scores using a different numbers of principal components
	for nc in range(1,21) + [30,40,50]:
		
		feature_name_nc = feature_name +"_{0}c".format(nc)
		specs[feature_name_nc] = dict()
		
		#Emulator
		specs[feature_name_nc]["emulator"] = emulator_pca.features({feature_name:range(nc)})
		specs[feature_name_nc]["emulator"] = specs[feature_name_nc]["emulator"].combine_features({feature_name_nc:[feature_name]})
		
		#Data and covariance
		pow_nc = pow[range(nc)]
		pow_nc.add_name(feature_name_nc)
		specs[feature_name_nc]["data"] = pow_nc.head(1600).mean()
		specs[feature_name_nc]["data_covariance"] = pow_nc.cov() / 1600

	#####################################################################################################################################

	#Compute scores projecting on (Om,w) natural directions
	projected_feature_name = feature_name + "_Om_w"
	specs[projected_feature_name] = dict()

	#Emulator
	specs[projected_feature_name]["emulator"] = emulator_pca.refeaturize(lambda f:f[feature_name].project(v,names=["vOm","vw"]),method="apply_whole").combine_features({projected_feature_name:[feature_name]})

	#Data and covariance
	pow_projected = pow.project(v,names=["vOm","vw"])
	pow_projected.add_name(projected_feature_name)
	specs[projected_feature_name]["data"] = pow_projected.head(1600).mean()
	specs[projected_feature_name]["data_covariance"] = pow_projected.cov() / 1600

	#####################################################################################################################################

	#Compute scores projecting on (Om,w) natural directions, and 3 random directions dictated by cosmic variance
	projected_feature_name = feature_name + "_Om_w_noise_nn{0}".format(n_noise)
	specs[projected_feature_name] = dict()

	#Emulator
	specs[projected_feature_name]["emulator"] = emulator_pca.refeaturize(lambda f:f[feature_name].project(v+tuple([dn_pca.iloc[nn] for nn in range(n_noise)]),names=["vOm","vw"]+["n{0}".format(nn) for nn in range(n_noise)]),method="apply_whole").combine_features({projected_feature_name:[feature_name]}).features({projected_feature_name:["vOm","vw"]})

	#Data and covariance
	pow_projected = pow.project(v+tuple([dn_pca.iloc[nn] for nn in range(n_noise)]),names=["vOm","vw"]+["n{0}".format(nn) for nn in range(n_noise)])[["vOm","vw"]]
	pow_projected.add_name(projected_feature_name)
	specs[projected_feature_name]["data"] = pow_projected.head(1600).mean()
	specs[projected_feature_name]["data_covariance"] = pow_projected.cov() / 1600


#Execute
chi2database(db_name="../data/scores/power_cross_scores_LSST_Om-w.sqlite",parameters=score_parameters,specs=specs,pool=None,nchunks=None,table_name="survey_mean_PCA")