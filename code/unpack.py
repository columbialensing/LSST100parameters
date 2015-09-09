#!/usr/bin/env python-mpi
import os

from lenstools.utils import MPIWhirlPool
from lenstools.pipeline.simulation import SimulationBatch


#MPIPool
try:
	pool = MPIWhirlPool()
except:
	pool = None

#Current simulation batch 
batch = SimulationBatch.current()
batch.unpack(where=batch.environment.storage,pool=pool)
pool.comm.Barrier()
