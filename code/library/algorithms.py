from lenstools.statistics.ensemble import Ensemble,Series
from lenstools.statistics.constraints import Emulator
import numpy as np


#############################################################################
#Study the effect on a particular feature when we vary a parameter at a time#
#############################################################################

class Manifold(object):

	def __init__(self,emulator):
		
		assert isinstance(emulator,Emulator)
		self.emulator = emulator
		self.feature_names = emulator.feature_names
		
		#Compute the scale on each bin, and compute the PCA in feature space
		scale = self.emulator[self.feature_names].mean()
		self.pca = self.emulator[self.feature_names].principalComponents(location=scale,scale=scale)

	#Find where the feature is pointing in the PCA coordinate space
	def transform(self,point={"Om":0.26,"w":-1,"sigma8":0.8}):
		parameters = parameters = point.keys()
		s = Series([point[p] for p in parameters],index=parameters)
		return self.pca.transform(self.emulator.predict(s[self.emulator.parameter_names])) 

	#Make three lines in parameter space, varying one parameter at a time
	def draw_tangents(self,start={"Om":0.26,"w":-1,"sigma8":0.8},end={"Om":0.29,"w":-0.8,"sigma8":1.0},npoints=100,tangents=True):
	
		parameters = start.keys()
		npar = len(parameters)
		line = dict()

		#One Ensemble for every line
		for p in parameters:
			line[p] = Ensemble(np.zeros((npoints,npar)),columns=parameters)
		
			#Fill in with start value
			for q in parameters:
				line[p][q] = start[q]

			#Vary one parameter at a time
			line[p][p] = np.linspace(start[p],end[p],npoints)


		#Compute the pca components along the single parameter lines
		components = dict()
	
		for p in parameters:
			feature_on_line = self.emulator.predict(line[p][self.emulator.parameter_names])
			components[p] = self.pca.transform(feature_on_line)

		#Return the PCA components along the one parameter variation directions
		return Curve(components,tangents=tangents)

	#Make lines in parameter space that correspond to nuisance parameters
	def draw_nuisance(self,start,end,func,npoints=100,tangents=True,base={"Om":0.26,"w":-1,"sigma8":0.8}):

		nuisance = start.keys()
		parameters = base.keys()
		components = dict()

		#Unperturbed features
		line = Ensemble(np.zeros((npoints,len(parameters))),columns=parameters)
		for p in parameters:
			line[p] = base[p]
		feature_on_line = self.emulator.predict(line[self.emulator.parameter_names])

		#Emulate the unperturbed features, then add the perturbation due to the nuisance parameters
		for n in nuisance:
			feature_on_line[n] = np.linspace(start[n],end[n],npoints)
			nuisance_features = feature_on_line.apply(lambda r:func[n](r[self.feature_names],r[(n,"")]),axis=1)
			components[n] = self.pca.transform(nuisance_features)

		#Return
		return Curve(components,tangents=tangents)

	#Draw a grid of projected features
	def draw_grid(self,parameters,projection_vectors,names=None):

		#Emulate the features and project
		emulated_features = self.emulator.predict(parameters)
		projected_features = self.pca.transform(emulated_features).project(projection_vectors,names)

		#return to user
		return Grid(Ensemble.concat((parameters,projected_features),axis=1))

###################################################################################################################

class Curve(object):

	def __init__(self,components,tangents):
		self.components = components
		if tangents:
			self.tangents()

	def __getitem__(self,n):
		return self.components[n]

	@property 
	def directions(self):
		return self.components.keys()

	@property 
	def ncol(self):
		return len(self[self.directions[0]].columns)

	@property 
	def feature_names(self):
		return self[self.directions[0]].columns

	#Normalize
	def tangents(self):
		for p in self.components.keys():
			self.components[p] = self.components[p].diff().apply(lambda r:r/np.sqrt((r**2).sum()),axis=1)
	
	#Dot two directions together
	def dot(self,v1,v2,other=None):
		
		if other is None:
			other = self
		
		return self[v1[0]].iloc[v1[1]].dot(other[v2[0]].iloc[v2[1]])

	#Project on a hyperplane defined by a tuple of linearly independent vectors; returns components along these vectors
	def project(self,v,names=None):
		return self.__class__(dict((p,self[p].project(v,names)) for p in self.directions),tangents=False)

###################################################################################################################

class Grid(object):

	def __init__(self,points):
		self.points = points

	def plot(self):
		pass

