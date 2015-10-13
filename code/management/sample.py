#!/usr/bin/env python

from lenstools.simulations import Design
import numpy as np

#Create design
d = Design()

#Add parameters
d.add_parameter("Om",0.2,0.5,"Om")
d.add_parameter("w",-1.5,-0.5,"w")
d.add_parameter("si8",0.5,1.2,"si8")

#Lay down points
d.put_points(100)

#Sample
d.sample(Lambda=1.0,p=2.0,seed=123,maxIterations=1000000)
p = d.points

#Save in increasing sigma8 order
np.save("../data/design.npy",p[p[:,2].argsort()])
