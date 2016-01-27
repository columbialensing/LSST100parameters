#Default settings for feature analysis (typically not to be touched)

import numpy as np

#Settings for Fisher matrix
class FisherSettings(object):

	#Constructor
	def __init__(self):

		#Cosmological parameters and available features
		features = ["power_spectrum","moments","peaks"]
		
		#Power spectrum
		self.power_spectrum = FeatureSettings("cross_spectra.sqlite","means","features_fiducial","features_fiducial_EmuIC","models")
		self.power_spectrum.set_feature_labels(["l{0}".format(n) for n in range(1,51)])
		self.power_spectrum.set_redshift_labels(["b1","b2"])
		
		#Moments
		self.moments = FeatureSettings("moments.sqlite","means","features_fiducial","features_fiducial_EmuIC","models")
		self.moments.set_feature_labels(["sigma0","sigma1","S0","S1","S2","K0","K1","K2","k3"])
		self.moments.set_redshift_labels(["b1"])
		
		#Peaks
		self.peaks = FeatureSettings("peaks.sqlite","means","features_fiducial","features_fiducial_EmuIC","models")
		self.peaks.set_feature_labels(["k{0}".format(n) for n in range(1,51)])
		self.peaks.set_redshift_labels(["b1"])

#Settings for each feature type
class FeatureSettings(object):

	#Constructor
	def __init__(self,dbname,emulator_table,covariance_table,data_table,model_table):
		
		#Typically not touched
		self.dbname = dbname
		self.emulator_table = emulator_table
		self.covariance_table = covariance_table
		self.data_table = data_table
		self.model_table = model_table

	#Set feature label names
	def set_feature_labels(self,names):
		self.feature_labels = names

	#Set redshift indices names
	def set_redshift_labels(self,names):
		self.redshift_labels = names

	##################################
	############Query formats#########
	##################################

	#Data and covariance query
	def data_query(self,feature_filter=None,redshift_filter=None,realization_filter=None):

		#Build the query on top of this
		query = "SELECT realization,"

		#Feature columns
		if feature_filter is not None:
			self.feature_labels = feature_filter.split(",")
		query += ",".join(self.feature_labels)

		#Redshift columns
		query += "," + ",".join(self.redshift_labels)
		
		#Table name
		query += " FROM {0}".format(self.data_table)

		#Additional redshift filter
		if (redshift_filter is not None) or (realization_filter is not None):
			query += " WHERE "

		if redshift_filter is not None:
			query += redshift_filter

		if realization_filter is not None:
			if redshift_filter is None:
				query += realization_filter
			else:
				query += " AND "+realization_filter

		#Return to user
		return query

	def covariance_query (self,feature_filter=None,redshift_filter=None,realization_filter=None):
		return self.data_query(feature_filter,redshift_filter,realization_filter).replace("FROM {0}".format(self.data_table),"FROM {0}".format(self.covariance_table))

	#Emulator query
	def emulator_query(self,feature_filter=None,redshift_filter=None):
		data_query = self.data_query(feature_filter,redshift_filter,None)
		emulator_query = data_query.replace("SELECT realization,","SELECT model,")
		return emulator_query.replace("FROM {0}".format(self.data_table),"FROM {0}".format(self.emulator_table)) 
	

#Instance of default analysis settings
settings = FisherSettings()
