#!/usr/bin/env python

import sys

import lenstools

from lenstools.pipeline.simulation import SimulationBatch,LensToolsCosmology
from lenstools.pipeline.settings import EnvironmentSettings,NGenICSettings
from lenstools.simulations.camb import CAMBSettings
from lenstools.simulations.gadget2 import Gadget2Settings

import numpy as np

#Settings
camb = CAMBSettings()
ngenic = NGenICSettings()
gadget2 = Gadget2Settings()

zmax = 3.1
nlenses = 60

#NGenIC
ngenic.GlassFile = lenstools.data("dummy_glass_little_endian.dat")

#Gadget
gadget2.NumFilesPerSnapshot = 24

#Init batch
environment = EnvironmentSettings(home="/Users/andreapetri/Documents/Columbia/Simulations/LSST100parameters/Test/Home",storage="/Users/andreapetri/Documents/Columbia/Simulations/LSST100parameters/Test/Storage")
batch = SimulationBatch(environment)

if sys.argv[1]=="--tree":

	#Add all the models,collections and one realization
	seed = np.random.randint(10000000)

	p = np.load("../data/design.npy")
	d = list()

	for Om,w,si8 in p:
	
		#Lay down directory tree
		cosmo = LensToolsCosmology(Om0=Om,Ode0=1-Om,w0=w,sigma8=si8)
		model = batch.newModel(cosmo,parameters=["Om","Ol","w","si"])
		collection = model.newCollection(box_size=260.0*model.Mpc_over_h,nside=512)
		r = collection.newRealization(seed)


if sys.argv[1]=="--camb":

	#CAMB settings
	for model in batch.available:
		collection = model.collections[0]
		collection.writeCAMB(z=np.array([0.0]),settings=camb)


if sys.argv[1]=="--pfiles":

	#Compute comoving distance to maximum redshift for each model
	d = list()
	for model in batch.available:
		d.append(model.cosmology.comoving_distance(zmax))

	#Compute lens spacings
	d = np.array([dv.value for dv in d]) * d[0].unit

	for model in batch.available:

		collection = model.collections[0]
		r = collection.realizations[0]

		#ngenic parameter file
		r.writeNGenIC(ngenic)

		#Compute the redshifts of the Gadget snapshots

		#Gadget parameter file
		r.writeGadget2(gadget2)