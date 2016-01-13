Docker containers for the test run
==================================

These containers should be run in [the docker image that we provided](https://hub.docker.com/r/apetri/lenstools-ubuntu/).   

0. Generation of the initial conditions
---------------------------------------
This preliminary step generates the initial condition files that will then be evolved in step 1. All the input files necessary for step 0 are present in the Docker image that we provided. The command that needs to be run is described in _0-ngenic.sh_. This step will generate approximately 4GB of output.

1. Temporal evolution of the initial conditions
-----------------------------------------------
This step will take in the initial condition files from step 0 as an input an evolve them using a publicly available code called _Gadget2_ (that is already installed in the docker image).
This is the most expensive step in terms of CPU hours. To run this container, the command looks like the one described in
_1-gadget.sh_. This step will take approximately 23 hours to complete, and will produce approximately 250GB of output. 

2. Two dimensional slicing of the simulation volumes
----------------------------------------------------
This step performs a data reduction on the output of step 1 by reading in each simulation cube (called a _snapshot_) at different
times, and slicing it into two dimensional slabs for the final processing that takes place in step 3. This step will reduce the total
output disk space from 250GB (full 3D information) to 34GB (information in the projected 2D slabs). To run this step you need to issue the command
contained in _2-planes.sh_. In principle it is possible to combine step 1 and step 2 in a single step to avoid writing the intermediate 250GB to disk. 
To do this, the current code implementation requires a machine with at least 32 cores and 128GB of memory. 
