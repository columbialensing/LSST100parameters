#!/usr/bin/env python

import sys,os
sys.modules["mpi4py"] = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.simulations import PotentialPlane

import astropy.units as u

#Get handle on the plane set
batch = SimulationBatch.current()
model = batch.available[0]
plane_set = model.collections[0].realizations[0].planesets[0]

#Visualize each plane
for n in range(60):

	plane_file = plane_set.path("snap{0}_potentialPlane0_normal0.fits".format(n))
	print("[+] Processing {0}".format(plane_file))

	plane = PotentialPlane.load(plane_file).density()
	plane = plane.smooth(10.0*u.Mpc,kind="gaussianFFT")
	plane.visualize()
	plane.ax.set_title(r"$t={0:.3f}$".format(plane_set.cosmology.age(plane.redshift).value)+r"$\mathrm{Gyr}$",fontsize=18)

	figure_filename = os.path.join(batch.environment.home,"planes_png","plane{0}.png".format(n))
	print("[+] Saving figure to {0}".format(figure_filename))
	plane.savefig(figure_filename)

	plt.close(plane.fig)

