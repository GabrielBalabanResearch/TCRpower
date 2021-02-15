import numpy as np 
import numdifftools as nd
from scipy import stats
from scipy import optimize
from functools import partial

class TCRPowerCalculator:
	def __init__(self, pcmodel):
		self.pcmodel = pcmodel
		self.predict_variance = self.pcmodel.predict_variance
		self.predict_mean = self.pcmodel.predict_mean
		self.get_prediction_interval = self.pcmodel.get_prediction_interval
		self.predict_detection_probability = self.pcmodel.predict_detection_probability

	#possivle TODO: Parse this method out into a new 2-step model class
	def predict_detection_probability_2step(self, tcr_frequency, num_reads, num_cells, detect_thresh = 1):		
		"""
		2-step detection probability model where 
		
		1) Num_cells_TCR is sampled first from the blood (Poisson model)
		2) The RNA detection probability is calculated (Negbin model).
		
		The num_cells_TCR is marginalized with the num_cells parameter as the upper limit 
		on the number of cells that could be sampled for a given TCR.
		"""

		mu_cells = tcr_frequency*num_cells
		p0_poisson = stats.poisson.pmf(0, mu_cells)
		
		num_cells_TCR = np.arange(1, num_cells + 1)[:,np.newaxis]
		
		#Step 1 Poisson
		p1 = stats.poisson.pmf(num_cells_TCR, mu_cells)

		#Get rid of 0 probability cell counts
		num_cells_TCR = num_cells_TCR[p1 >0]
		p1 = p1[p1 >0]

		#Step 2 Negbin
		mu_reads = self.pcmodel.predict_mean(num_cells_TCR/num_cells, num_reads)
				
		p2 = np.zeros(p1.shape)
		for i in np.arange(detect_thresh):
			p2 += self.pcmodel.pmf(mu_reads, count = i)

		p0_2step = np.dot(p1.squeeze(), p2.squeeze())

		#If 0 cells from Poisson model then automatically get 0 reads
		return 1.0 - p0_poisson - p0_2step
	
	def get_limit_of_detection_tcrfreq(self, num_reads, conf_level = 0.95):
		opt_f = partial(self.pcmodel.predict_detection_probability, num_reads = num_reads) 

		opt_res = optimize.root_scalar(lambda freq: opt_f(freq) - conf_level,
		 								method = "brentq",
		 								bracket = [1.0e-16, 1])
		return opt_res.root

	def get_limit_of_detection_nreads(self, tcr_freq, conf_level = 0.95):
		opt_nreads = partial(self.pcmodel.predict_detection_probability, tcr_frequencies = tcr_freq) 

		opt_res = optimize.root_scalar(lambda nreads: opt_nreads(num_reads = nreads) - conf_level,
										method = "secant",
										x0 = 1.0e-16,
										x1 = 1)
		
		return int(np.around(opt_res.root))