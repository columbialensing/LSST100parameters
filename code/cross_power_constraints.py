#!/usr/bin/env python-mpi

import numpy as np

from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.constraints import Emulator
from library.featureDB import FeatureDatabase
import library.driver_mpi as driver

multipole_names = ["l{0}".format(n) for n in range(1,51)]
score_parameters = Ensemble.meshgrid({"Om":np.linspace(0.2,0.5,50),"w":np.linspace(-1.2,-0.8,50)})
score_parameters["sigma8"] = 0.8

#Specifications
specs = dict()

specs["features"] = dict()
specs["features"]["emulator"] = Emulator.read("../models/emulator_cross_11.pkl")

with FeatureDatabase("../models/cross_spectra_fiducial.sqlite") as db:
	pow = db.query("SELECT {0} FROM features_fiducial WHERE b1=1 AND b2=1".format(",".join(multipole_names)))
	pow.add_name("features")
	specs["features"]["data"] = pow.mean()
	specs["features"]["data_covariance"] = pow.cov() 

#Execute
driver.chi2database(db_name="../models/cross_spectra_scores.sqlite",parameters=score_parameters,specs=specs,pool=None,nchunks=None)