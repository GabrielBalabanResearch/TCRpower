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

	#possivle TODO: Parse this method out into a new 2-step model class
	def predict_detection_probability_2step(self, tcr_frequencies, num_reads, num_cells):
		"""
		2-step detection probability model where 
		
		1) Num_cells_TCR is sampled first from the blood (Poisson model)
		2) The RNA detection probability is calculated (Negbin model).
		
		The num_cells_TCR is marginalized with the num_cells parameter as the upper limit 
		on the number of cells that could be sampled for a given TCR.
		"""

		mu_cells = tcr_frequencies*num_cells
		p0_poisson = stats.poisson.pmf(0, mu_cells)
		
		num_cells_TCR = np.arange(1, num_cells + 1)[:,np.newaxis]

		#Step 1 Poisson
		p1 = stats.poisson.pmf(num_cells_TCR, mu_cells)

		#Step 2 Negbin
		
		mu_reads = self.pcmodel.predict_mean(num_cells_TCR/num_cells, num_reads)
		p2 = self.pcmodel.pmf(mu_reads, count = 0)
		
		#alpha = self.pcmodel.alpha
		#n,p = rp_negbin_params(alpha, mu_reads)
		#p2 = stats.nbinom.pmf(0, n, p)

		p0_2step = (p2*p1).sum(axis = 0)

		return 1.0 - p0_poisson - p0_2step
	
	def get_limit_of_detection_tcrfreq(self, num_reads, conf_level = 0.95):
		opt_f = partial(self.pcmodel.predict_detection_probability, num_reads = num_reads) 

		opt_res = optimize.root_scalar(lambda freq: opt_f(freq) - conf_level,
										method = "brentq",
										bracket = [0,1])
		return opt_res.root

	def get_limit_of_detection_nreads(self, tcr_freq, conf_level = 0.95):
		opt_nreads = partial(self.pcmodel.predict_detection_probability, tcr_frequencies = tcr_freq) 

		opt_res = optimize.root_scalar(lambda nreads: opt_nreads(num_reads = nreads) - conf_level,
										method = "secant",
										x0 = 0,
										x1 = 1)
		
		return int(np.around(opt_res.root))