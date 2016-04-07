#!/usr/bin/env python-mpi
import os
import argparse

from lenstools.utils import MPIWhirlPool
from lenstools.pipeline.simulation import SimulationBatch

#Get the resource to compress
def shear(model,batch,search_string):
	full_path =  model[search_string].storage
	rel_path = os.path.relpath(full_path,batch.environment.storage)
	return rel_path

#MPIPool
try:
	pool = MPIWhirlPool()
except:
	pool = None

#Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e","--environment",dest="env_file",action="store",type=str,default="environment.ini",help="environment option file")
parser.add_argument("-a","--archive",dest="archive",action="store",type=str,default="archive/{0}.tar.gz",help="archive name format")
parser.add_argument("-s","--search",dest="search",action="store",type=str,default="c0C0",help="search string that when dialed points to the resource")

cmd_args = parser.parse_args()

#Current simulation batch 
batch = SimulationBatch.current(cmd_args.env_file)
models = batch.models
archive_names = [cmd_args.archive.format(m.cosmo_id) for m in models]

batch.archive(archive_names,pool=pool,resource=shear,chunk_size=1,batch=batch,search_string=cmd_args.search)


pool.comm.Barrier()
