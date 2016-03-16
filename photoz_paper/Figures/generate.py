#!/usr/bin/env python

import sys,argparse
sys.modules["mpi4py"] = None

import numpy as np
import matplotlib.pyplot as plt

from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.database import Database


###################################################################################################


###########################################################################################################################################

#Method dictionary
method = dict()

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()