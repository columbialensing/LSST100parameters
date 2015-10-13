LSST100Parameters simulation batch
======

##How to import this simulation batch on your own machine

After cloning this repository, create inside it a file called _environment.ini_, open it and put the following contents in it

	[EnvironmentSettings]
	
	home = /path/to/your/home/portion/of/the/batch 
	storage = /path/to/your/storage/portion/of/the/batch 

After this you should be able to get a handle of the simulations with the lenstools SimulationBatch class

	from lenstools.pipeline import SimulationBatch
	batch = SimulationBatch.current()

The _home_ part (the light part of the batch) should now be available to you; the _storage_ part (i.e. the heavy part, with all the simulated maps,catalogs, etc...) you will have to obtain from us. 

##How to compress/unpack the simulation products


###Compress

You should use the _management/compress.py_ script: you can tweak the _shear_ function to select which resource in the simulation batch you want to compress, the _models_ variable to select which models you want to compress and the _archive_names_ variable to tell the script where the compress archives should be put (these paths will be considered as relative to the storage part of the simulation batch). Then run the script in parallel

	mpiexec -n <len(models)> ./compress.py

###Unpack

You should use the _management/unpack.py_ script: you have to tweak the _where=_ keyword in _batch.unpack_ method to tell the script where the gzipped archives are (not that you should be already in possession of the home part of this simulation set by cloning this repository). Then run the script in parallel

	mpiexec -n <number_of_archives> ./unpack.py
