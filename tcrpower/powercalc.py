import numpy as np 

class PowerCalculator:
	def __init__(self, pcmodel):
		self.pcmodel = pcmodel

	def get_variance(self, TCR_frequency, Nread):
		pread = self.pcmodel.pread
		alpha = self.pcmodel.alpha
		mu = TCR_frequency*pread*Nread
		
		return mu + alpha*mu**2