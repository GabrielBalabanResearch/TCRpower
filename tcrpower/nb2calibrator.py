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
		
		eta: Variance scaling parameter

	Negative Binomial Parameterization is the NB2 model with mean mu 
	and variance
	
		var = mu + eta*mu^2
	
	The density function for a single data-point is then
	.. math::
		f(Y,r,p) = \frac{\gamma(Y + r)}{\gamma(Y + 1) \gamma(r)} (1 -p)^Y p^r
		f(Y, eta, mu) = \frac{\gamma(Y + eta^-1)}{\gamma(Y + 1) \gamma(eta^-1)} (\frac{\eta \mu}{1 + \eta \mu})^Y (1 + \eta \mu)^(-eta^-1)
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
		#Get the initial sread_eff from a Poisson model
		eta0 = 0.001

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			poisson_fam = sm.families.Poisson(link = sm.genmod.families.links.identity())
			fread0 = sm.GLM(self.C, self.fmix*self.Nread, poisson_fam).fit().params
		return np.hstack([fread0, eta0])

	def llh(self, read_eff, eta):
		mu = self.fmix*read_eff*self.Nread

		etainv = eta**-1
		llh = gammaln(self.C + etainv) - gammaln(etainv) - gammaln(self.C +1)
		llh += self.C*np.log(eta*mu) - (self.C + etainv)*np.log(1 + eta*mu)
		return np.sum(llh)

	#First derivative functions
	def score(self, read_eff, eta):
		mu = self.fmix*read_eff*self.Nread
		
		#eta derivative
		dllh_da = self.score_eta(eta, mu)		

		#fread derivative
		dllh_dfread = self.Nread*self.fmix.T @ self.dllh_dmu(eta, mu)

		score = np.hstack([dllh_dfread, dllh_da.sum()])
		return score

	def score_eta(self, eta, mu):
		etainv = 1.0/eta
		etamu = eta*mu

		dllh_da = etainv**2*(digamma(etainv) - digamma(self.C + etainv))
		dllh_da += self.C*etainv
		dllh_da += etainv**2*np.log(1 + etamu) - mu*(self.C + etainv)/(1 + etamu)
		return dllh_da

	def dllh_dmu(self, eta, mu):
		return self.C/mu - (1 + self.C*eta)/(1 + mu*eta)

#Helper functions to switch between negbin parameterizations (NB2 model)
def rp_negbin_params(eta, mu):
	r = 1.0/eta
	p = 1/(1 + mu*eta)
	return r,p

def eta_mu_negbin_params(r, p):
	eta = 1.0/r
	mu = (1 - p)*r/p
	return eta, mu

class NB2TCRCountModel:
	"Parameterized Negative Binomial 2 model"
	def __init__(self, read_eff, eta):
		self.read_eff = read_eff
		self.eta = eta

	def predict_mean(self, tcr_frequencies, num_reads):
		return tcr_frequencies*self.read_eff*num_reads

	def predict_variance(self, tcr_frequencies, num_reads):
		mu = self.predict_mean(tcr_frequencies, num_reads)
		return mu + self.eta*mu**2

	def pmf(self, mu, count = 0):
		eta = self.eta
		r,p = rp_negbin_params(eta, mu)
		return stats.nbinom.pmf(count, r, p)

	def predict_detection_probability(self, tcr_frequencies = 1.0, num_reads = 1, detect_thresh = 1):
		"""
		Models detection probability with negative binomial models assuming RNA receptor frequencies
		detect_thresh = Minimum number of reads before a TCR is considered "detected".
		"""
		
		#TODO: Implement a detection probability threshold by summing over the first argument of the pmf.
		mu = self.predict_mean(tcr_frequencies, num_reads)
		p_belowthresh = self.pmf(mu, count = np.arange(detect_thresh)[:,np.newaxis]).sum(axis =0)
		return 1.0 - p_belowthresh

	def get_prediction_interval(self, tcr_frequencies, num_reads, interval_size = 0.95):
		eta = self.eta
		mu = self.predict_mean(tcr_frequencies, num_reads)
		r,p = rp_negbin_params(eta, mu)
		return stats.nbinom.interval(interval_size, r, p)
