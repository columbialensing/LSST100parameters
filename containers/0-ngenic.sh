#!/bin/bash

mpiexec -n 32 /N-body/NGenIC/NGenIC /TestRun/models/Om0.260_Ol0.740_w-1.000_si0.800/512b260/ic1/ngenic.param > ngenic.out 2> ngenic.err
