#!/bin/bash

mpiexec -n 16 lenstools.planes-mpi -e /TestRun/environment.ini -c /TestRun/planes.ini -v 'Om0.260_Ol0.740_w-1.000_si0.800|512b260|ic1' > planes.out 2> planes.err