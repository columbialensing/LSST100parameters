#!/usr/bin/env python-mpi
import os

from lenstools.utils import MPIWhirlPool
from lenstools.pipeline.simulation import SimulationBatch

#Get the resource to compress
def shear(model,batch):
	full_path =  model.collections[0].getCatalog("Shear").storage_subdir
	rel_path = os.path.relpath(full_path,batch.environment.storage)
	return rel_path

#MPIPool
try:
	pool = MPIWhirlPool()
except:
	pool = None

#Current simulation batch 
batch = SimulationBatch.current()
models = batch.models
archive_names = ["archive/{0}.tar.gz".format(m.cosmo_id) for m in models]

batch.archive(archive_names,pool=pool,resource=shear,chunk_size=1,batch=batch)


pool.comm.Barrier()
