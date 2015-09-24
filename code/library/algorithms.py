from lenstools.statistics.ensemble import Ensemble,Series
from lenstools.statistics.constraints import Emulator
import numpy as np


#############################################################################
#Study the effect on a particular feature when we vary a parameter at a time#
#############################################################################

class Line(object):

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
	def draw(self,start={"Om":0.26,"w":-1,"sigma8":0.8},end={"Om":0.29,"w":-0.8,"sigma8":1.0},npoints=100,normalize=True):
	
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
		return Components(components,normalize=normalize)

	#Make lines in parameter space that correspond to nuisance parameters
	def draw_nuisance(self,start,end,func,npoints=100,normalize=True,base={"Om":0.26,"w":-1,"sigma8":0.8}):

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
		return Components(components,normalize=normalize)


class Components(object):

	def __init__(self,components,normalize):
		self.components = components
		if normalize:
			self.normalize()

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
	def normalize(self):
		for p in self.components.keys():
			self.components[p] = self.components[p].apply(lambda c:c-c[0],axis=0)
			self.components[p] = self.components[p].apply(lambda r:r/np.sqrt((r**2).sum()),axis=1)
	
	#Dot two directions together
	def dot(self,v1,v2,other=None):
		
		if other is None:
			other = self
		
		return self[v1[0]].iloc[v1[1]].dot(other[v2[0]].iloc[v2[1]])

	#Project on a plane defined by b1 and b2; returns components along b1 and b2
	def project(self,b1,b2,names=None):
		
		#Build projection matrix
		projection_matrix = Ensemble(np.zeros((self.ncol,2)),columns=names,index=self.feature_names)
		columns = projection_matrix.columns
		cos_angle = b1.dot(b2)
		projection_matrix[columns[0]] = (b1 - b2*cos_angle) / (1 - cos_angle**2)
		projection_matrix[columns[1]] = (b2 - b1*cos_angle) / (1 - cos_angle**2)

		#Compute projections
		return self.__class__(dict((p,self[p].dot(projection_matrix)) for p in self.directions),normalize=False)

