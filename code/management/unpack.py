#!/usr/bin/env python-mpi
import os
import argparse

from lenstools.utils import MPIWhirlPool
from lenstools.pipeline.simulation import SimulationBatch


#MPIPool
try:
	pool = MPIWhirlPool()
except:
	pool = None

#Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e","--environment",dest="env_file",action="store",type=str,default="environment.ini",help="environment option file")
parser.add_argument("-w","--where",dest="where",action="store",type=str,default="/scratch3/scratchdirs/apetri/archive",help="path to the directory with the cosmo_id.tar.gz archives")

cmd_args = parser.parse_args()

#Current simulation batch 
batch = SimulationBatch.current(cmd_args.env_file)
batch.unpack(where=cmd_args.where,pool=pool)
pool.comm.Barrier()