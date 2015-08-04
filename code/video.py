#!/usr/bin/env python

import sys
sys.modules["mpi4py"] = None

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.simulations import PotentialPlane

batch = SimulationBatch.current()
model = batch.available[0]
print(model)