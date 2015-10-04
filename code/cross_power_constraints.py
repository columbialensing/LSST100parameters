#!/usr/bin/env python-mpi

import numpy as np

from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.constraints import Emulator
from library.featureDB import FeatureDatabase
from library.algorithms import Manifold
import library.driver_mpi as driver

#Score parameters
multipole_names = ["l{0}".format(n) for n in range(1,51)]
score_parameters = Ensemble.meshgrid({"Om":np.linspace(0.24,0.27,100),"w":np.linspace(-1.1,-1.0,100)})
score_parameters["sigma8"] = 0.8

#Specifications
specs = dict()

#Open the fiducial cosmology database for data and covariances
with FeatureDatabase("../models/cross_spectra_fiducial.sqlite") as db:

	#Process all the five power spectrum bins
	for n in range(5):

		#Name the feature
		feature_name = "power_cross_{0}{0}".format(n)
		specs[feature_name] = dict()

		#Add the emulator
		emulator = Emulator.read("../models/emulator_power_cross_{0}{0}.pkl".format(n))
		manifold = Manifold(emulator)
		curves = manifold.draw_tangents()
		v = (curves['Om'].iloc[1],curves['w'].iloc[1])

		#Refeaturize the emulator
		specs[feature_name]["emulator"] = emulator.refeaturize(lambda f:manifold.pca.transform(f).project(v,names=['vOm','vw']),method="apply_whole")
		specs[feature_name]["emulator"].to_pickle("../models/emulator_power_cross_projected_{0}{0}.pkl".format(n)) 


		#Get the data and compute the covariance matrix
		pow = db.query("SELECT {0} FROM features_fiducial WHERE b1={1} AND b2={1}".format(",".join(multipole_names),n))
		pow.add_name(feature_name)
		pow = manifold.pca.transform(pow).project(v,names=['vOm','vw'])
		pow.add_name(feature_name)

		specs[feature_name]["data"] = pow.head(1600).mean()
		specs[feature_name]["data_covariance"] = pow.cov() / 1600

#Execute
driver.chi2database(db_name="../models/power_cross_scores_projected_scaled.sqlite",parameters=score_parameters,specs=specs,pool=None,nchunks=None)