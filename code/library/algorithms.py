from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.constraints import Emulator
import numpy as np


#############################################################################
#Study the effect on a particular feature when we vary a parameter at a time#
#############################################################################

class Line(object):

	def __init__(self,emulator):
		
		assert isinstance(emulator,Emulator)
		self.emulator = emulator
		
		#Compute the scale on each bin, and compute the PCA in feature space
		scale = self.emulator[self.emulator.feature_names].mean()
		self.pca = self.emulator[self.emulator.feature_names].principalComponents(location=scale,scale=scale)


	#Make three lines in parameter space, varying one parameter at a time
	def draw(self,start={"Om":0.26,"w":-1,"sigma8":0.8},end={"Om":0.29,"w":-0.8,"sigma8":1.0},npoints=100,order=["Om","w","sigma8"],normalize=True):
	
		npar = len(start.keys())
		line = dict()

		#One Ensemble for every line
		for p in order:
			line[p] = Ensemble(np.zeros((npoints,npar)),columns=order)
		
			#Fill in with start value
			for q in order:
				line[p][q] = start[q]

			#Vary one parameter at a time
			line[p][p] = np.linspace(start[p],end[p],npoints)


		#Compute the pca components along the single parameter lines
		components = dict()
	
		for p in line.keys():
			feature_on_line = self.emulator.predict(line[p])
			components[p] = self.pca.transform(feature_on_line)

			if normalize:
				components[p] = components[p].apply(lambda c:c-c[0],axis=0)
				components[p] = components[p].apply(lambda r:r/np.sqrt((r**2).sum()),axis=1)

		#Return the PCA components along the one parameter variation directions
		return Components(components)

class Components(object):

	def __init__(self,components):
		self.components = components
	
	#Dot two vectors together
	def dot(self,v1,v2):
		return self.components[v1[0]].iloc[v1[1]].dot(self.components[v2[0]].iloc[v2[1]])
