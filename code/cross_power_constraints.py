#!/usr/bin/env python-mpi

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
score_parameters = Ensemble.meshgrid({"Om":np.linspace(0.24,0.27,100),"w":np.linspace(-1.1,-1.0,100)})
score_parameters["sigma8"] = 0.8

#Specifications
specs = dict()

#Open the fiducial cosmology database for data and covariances
with FeatureDatabase("../data/features/cross_spectra_fiducial.sqlite") as db:

	#Process all the five power spectrum bins
	for n in range(5):

		#Name the feature
		feature_name = "power_cross_{0}{0}".format(n)
		projected_feature_name = feature_name + "_projected"
		specs[feature_name] = dict()
		specs[projected_feature_name] = dict()

		#Add the emulator
		emulator = Emulator.read("../data/emulators/emulator_power_cross_{0}{0}.pkl".format(n))
		specs[feature_name]["emulator"] = emulator

		#Projected version of the emulator
		manifold = Manifold(emulator)
		curves = manifold.draw_tangents()
		v = (curves['Om'].iloc[1],curves['w'].iloc[1])

		#Refeaturize the emulator
		emulator_projected = emulator.refeaturize(lambda f:manifold.pca.transform(f).project(v,names=['vOm','vw']),method="apply_whole")
		specs[projected_feature_name]["emulator"] = emulator_projected.combine_features({projected_feature_name:[feature_name]}) 

		#Get the data and compute the covariance matrix
		pow = db.query("SELECT {0} FROM features_fiducial WHERE b1={1} AND b2={1}".format(",".join(multipole_names),n))
		pow.add_name(feature_name)
		specs[feature_name]["data"] = pow.head(1600).mean()
		specs[feature_name]["data_covariance"] = pow.cov() / 1600

		#Do the same for the projected features
		pow = manifold.pca.transform(pow).project(v,names=['vOm','vw'])
		pow.add_name(projected_feature_name)
		specs[projected_feature_name]["data"] = pow.head(1600).mean()
		specs[projected_feature_name]["data_covariance"] = pow.cov() / 1600

#Execute
chi2database(db_name="../data/scores/power_cross_scores_LSST.sqlite",parameters=score_parameters,specs=specs,pool=None,nchunks=None)