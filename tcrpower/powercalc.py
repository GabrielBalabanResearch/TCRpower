import numpy as np 

class TCRPowerCalculator:
	def __init__(self, pcmodel):
		self.pcmodel = pcmodel

	def predict_mean(self, tcr_frequencies, num_reads):
		pread = self.pcmodel.pread
		return tcr_frequencies*pread*num_reads

	def predict_variance(self, tcr_frequencies, num_reads):
		alpha = self.pcmodel.alpha
		mu = self.predict_mean(tcr_frequencies, num_reads)
		return mu + alpha*mu**2