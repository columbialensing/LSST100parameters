#!/usr/bin/env python

import os
from lenstools.pipeline.simulation import SimulationBatch

#Current batch
batch = SimulationBatch.current()

#Purge snapshots
for model in batch.available:
	r = model.collections[0].realizations[0]
	os.system("rm {0}/snapshots/snapshot*".format(r.storage_subdir))
	print("[+] {0} purged".format(model.cosmo_id))