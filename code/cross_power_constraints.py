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
score_parameters = Ensemble.meshgrid({"Om":np.linspace(0.256,0.2635,100),"w":np.linspace(-1.08,-0.92,100)})
score_parameters["sigma8"] = 0.8

#Fiducial parameters
p_fiducial = Ensemble(np.array([0.26,-1.0,0.8])[None],columns=["Om","w","sigma8"])
p_fiducial.add_name("parameters")

#Specifications
specs = dict()

#Open the fiducial cosmology database for data and covariances
with FeatureDatabase("../data/features/cross_spectra_fiducial.sqlite") as db:

	#Name the feature
	feature_name = "cross_power"
	projected_feature_name = feature_name + "_projected"
	specs[feature_name] = dict()
	specs[projected_feature_name] = dict()

	#Add the emulator
	emulator = Emulator.read("../data/emulators/emulator_power_cross_all.pkl")
	specs[feature_name]["emulator"] = emulator

	#Get the data and compute the covariance matrix
	lab,pow = db.query("SELECT realization,b1,b2,{0} FROM features_fiducial".format(",".join(multipole_names))).suppress_indices(by=["realization"],suppress=["b1","b2"],columns=multipole_names)
	pow.pop("realization")
	pow.add_name(feature_name)

	#Compute mean and covariance
	specs[feature_name]["data"] = pow.head(1600).mean()
	specs[feature_name]["data_covariance"] = pow.cov() / 1600


#Execute
chi2database(db_name="../data/scores/power_cross_scores_LSST_Om-w.sqlite",parameters=score_parameters,specs=specs,pool=None,nchunks=None,table_name="survey_mean_with_fiducial")