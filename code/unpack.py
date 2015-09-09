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
batch.unpack(where='/scratch3/scratchdirs/apetri/archive',pool=pool)
pool.comm.Barrier()
