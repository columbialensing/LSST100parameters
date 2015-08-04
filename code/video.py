#!/usr/bin/env python

import sys
sys.modules["mpi4py"] = None

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.simulations import PotentialPlane

#Get handle on the plane set
batch = SimulationBatch.current()
model = batch.available[0]
plane_set = model.collections[0].realizations[0].planesets[0]

print(plane_set)
