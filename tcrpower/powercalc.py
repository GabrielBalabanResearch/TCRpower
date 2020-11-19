import numpy as np 
import numdifftools as nd
from tcrpower.calibrate import rp_negbin_params
from scipy import stats
from scipy import optimize

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

	def predict_detection_probability(self, tcr_frequencies, num_reads):
		"""
		Models detection probability with negative binomial models assuming RNA receptor frequencies
		"""
		alpha = self.pcmodel.alpha
		mu = self.predict_mean(tcr_frequencies, num_reads)
		r,p = rp_negbin_params(alpha, mu)
		return 1.0 - stats.nbinom.pmf(0, r, p)

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
		alpha = self.pcmodel.alpha

		mu_reads = self.predict_mean(num_cells_TCR/num_cells, num_reads)
		
		n,p = rp_negbin_params(alpha, mu_reads)
		p2 = stats.nbinom.pmf(0, n, p)

		p0_2step = (p2*p1).sum(axis = 0)

		return 1.0 - p0_poisson - p0_2step

	def get_prediction_interval(self, tcr_frequencies, num_reads, interval_size = 0.95):
		alpha = self.pcmodel.alpha
		mu = self.predict_mean(tcr_frequencies, num_reads)
		r,p = rp_negbin_params(alpha, mu)
		return stats.nbinom.interval(interval_size, r, p)

	def get_limit_of_detection_tcrfreq(self, num_reads, conf_level = 0.95):
		pread = self.pcmodel.pread
		alpha = self.pcmodel.alpha

		def opt_f(freq):
			mu = freq*pread*num_reads
			r, p = rp_negbin_params(alpha, mu)
			return 1 - stats.nbinom.pmf(0, r, p)

		opt_res = optimize.root_scalar(lambda freq: opt_f(freq) - conf_level,
										method = "brentq",
										bracket = [0,1])
		return opt_res.root

	def get_limit_of_detection_nreads(self, tcr_freq, conf_level = 0.95):
		pread = self.pcmodel.pread
		alpha = self.pcmodel.alpha

		def opt_nreads(num_reads):
			mu = tcr_freq*pread*num_reads
			r, p = rp_negbin_params(alpha, mu)
			return 1 - stats.nbinom.pmf(0, r, p)

		opt_res = optimize.root_scalar(lambda nreads: opt_nreads(nreads) - conf_level,
										method = "secant",
										x0 = 0,
										x1 = 1)
		
		return int(np.around(opt_res.root))