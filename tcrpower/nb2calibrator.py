import numpy as np 
import pandas as pd
import statsmodels.api as sm
import numdifftools as nd
import warnings
from scipy.special import gamma, gammaln, digamma, polygamma
from scipy import stats
from tcrpower.newtonfitter import NewtonFitter

class NB2Calibrator(object):
	"""
	Power Calculator calibration with Negative Binomial 2 model

	Performs calibration calculations for a detection power calculator
	using pilot/test data with known clonotype mixture frequencies 
	
	Parameters:
		  fmix = Mixture frequencies of TCR clonotypes
		  C = Measured count data for each TCR clonotype 
		  Nread = Total number of reads used during sequencing

	Output:
		A calibrated model with technical variance estimates that
		can be used for detection power calculations

		P_read : The proportion of reads which map
				 to the target known clonotypes.
		
		alpha: Variance scaling parameter

	Negative Binomial Parameterization is the NB2 model with mean mu 
	and variance
	
		var = mu + alpha*mu^2
	
	The density function for a single data-point is then
	.. math::
		f(Y,r,p) = \frac{\gamma(Y + r)}{\gamma(Y + 1) \gamma(r)} (1 -p)^Y p^r
		f(Y, alpha, mu) = \frac{\gamma(Y + alpha^-1)}{\gamma(Y + 1) \gamma(alpha^-1)} (\frac{\alpha \mu}{1 + \alpha \mu})^Y (1 + \alpha \mu)^(-alpha^-1)
	"""
	
	def __init__(self, tcr_frequencies, counts, num_reads):
		self.fmix = tcr_frequencies
		self.C = counts
		self.Nread = num_reads

	def fit(self, start_params = None,
				  stepsize = 1.0,
				  maxiter = 1000,
				  TOL = 1.0e-8,
				  show_convergence = False):

		if start_params is None:
			start_params = self.get_default_initparams()

		scorep = lambda p: self.score(p[0], p[1])
		llh_hess = nd.Jacobian(scorep, 1.0e-8)

		fitter = NewtonFitter(lambda p: self.llh(p[0], p[1]),
							  scorep,
							  llh_hess)
		
		fittingresult = fitter.fit(start_params = start_params,
								   stepsize = stepsize,
								   TOL = TOL,
								   maxiter = maxiter,
								   show_convergence = show_convergence)
		
		return NB2TCRCountModel(fittingresult.params[0], 
					    		fittingresult.params[1])

	def get_default_initparams(self):
		#Get the initial spread from a Poisson model
		alpha0 = 0.001

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			poisson_fam = sm.families.Poisson(link = sm.genmod.families.links.identity())
			fread0 = sm.GLM(self.C, self.fmix*self.Nread, poisson_fam).fit().params
		return np.hstack([fread0, alpha0])

	def llh(self, pread, alpha):
		mu = self.fmix*pread*self.Nread

		alphainv = alpha**-1
		llh = gammaln(self.C + alphainv) - gammaln(alphainv) - gammaln(self.C +1)
		llh += self.C*np.log(alpha*mu) - (self.C + alphainv)*np.log(1 + alpha*mu)
		return np.sum(llh)

	#First derivative functions
	def score(self, pread, alpha):
		mu = self.fmix*pread*self.Nread
		
		#alpha derivative
		dllh_da = self.score_alpha(alpha, mu)		

		#fread derivative
		dllh_dfread = self.Nread*self.fmix.T @ self.dllh_dmu(alpha, mu)

		score = np.hstack([dllh_dfread, dllh_da.sum()])
		return score

	def score_alpha(self, alpha, mu):
		alphainv = 1.0/alpha
		alphamu = alpha*mu

		dllh_da = alphainv**2*(digamma(alphainv) - digamma(self.C + alphainv))
		dllh_da += self.C*alphainv
		dllh_da += alphainv**2*np.log(1 + alphamu) - mu*(self.C + alphainv)/(1 + alphamu)
		return dllh_da

	def dllh_dmu(self, alpha, mu):
		return self.C/mu - (1 + self.C*alpha)/(1 + mu*alpha)

#Helper functions to switch between negbin parameterizations (NB2 model)
def rp_negbin_params(alpha, mu):
	r = 1.0/alpha
	p = 1/(1 + mu*alpha)
	return r,p

def alpha_mu_negbin_params(r, p):
	alpha = 1.0/r
	mu = (1 - p)*r/p
	return alpha, mu

class NB2TCRCountModel:
	"Parameterized Negative Binomial 2 model"
	def __init__(self, pread, alpha):
		self.pread = pread
		self.alpha = alpha

	def predict_mean(self, tcr_frequencies, num_reads):
		return tcr_frequencies*self.pread*num_reads

	def predict_variance(self, tcr_frequencies, num_reads):
		mu = self.predict_mean(tcr_frequencies, num_reads)
		return mu + self.alpha*mu**2

	def pmf(self, mu, count = 0):
		alpha = self.alpha
		r,p = rp_negbin_params(alpha, mu)
		return stats.nbinom.pmf(count, r, p)

	def predict_detection_probability(self, tcr_frequencies = 1.0, num_reads = 1, detect_thresh = 1):
		"""
		Models detection probability with negative binomial models assuming RNA receptor frequencies
		detect_thresh = Minimum number of reads before a TCR is considered "detected".
		"""
		
		#TODO: Implement a detection probability threshold by summing over the first argument of the pmf.
		mu = self.predict_mean(tcr_frequencies, num_reads)
		p_belowthresh = self.pmf(mu, count = np.arange(detect_thresh)).sum()
		return 1.0 - p_belowthresh

	def get_prediction_interval(self, tcr_frequencies, num_reads, interval_size = 0.95):
		alpha = self.alpha
		mu = self.predict_mean(tcr_frequencies, num_reads)
		r,p = rp_negbin_params(alpha, mu)
		return stats.nbinom.interval(interval_size, r, p)
