#!/bin/bash

lenstools.submission -e 'environment.ini' -j 'jobs.ini' -t 'ngenic' -s 'testcluster.DockerCluster' 'realizations.txt'
lenstools.submission -e 'environment.ini' -j 'jobs.ini' -t 'gadget2' -s 'testcluster.DockerCluster' 'realizations.txt'
lenstools.submission -e 'environment.ini' -j 'jobs.ini' -t 'planes' -o 'planes.ini' -s 'testcluster.DockerCluster' 'realizations.txt'
lenstools.submission -e 'environment.ini' -j 'jobs.ini' -t 'raytracing' -o 'catalog.ini' -s 'testcluster.DockerCluster' 'collections.txt' 