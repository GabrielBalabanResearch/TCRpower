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
		alpha = self.pcmodel.alpha
		mu = self.predict_mean(tcr_frequencies, num_reads)
		r,p = rp_negbin_params(alpha, mu)
		return 1.0 - stats.nbinom.pmf(0, r, p)

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
