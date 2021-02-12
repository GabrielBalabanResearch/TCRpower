import numpy as np 
import pandas as pd
import statsmodels.api as sm
import numdifftools as nd
import warnings
from scipy.special import gamma, gammaln, digamma, polygamma
from scipy import stats
from tcrpower.newtonfitter import NewtonFitter
from functools import partial
from scipy.optimize import minimize, minimize_scalar
from tcrpower.nb2calibrator import NB2Calibrator

class NBVarCalibrator(object):
	"""
	Power Calculator Calibrator

	Performs calibration calculations for a detection power calculator
	using pilot/test data with known clonotype mixture frequencies 
	
	Input:
		  fmix = Mixture frequencies of TCR clonotypes
		  C = Measured count data for each TCR clonotype 
		  Nread = Total number of reads used during sequencing

	Output:
		A calibrated model with technical variance estimates that
		can be used for detection power calculations

		P_read : The proportion of reads which map
				 to the target known clonotypes.
		
		alpha : variance scaling parameter
		lambda : variance exponential parameter

	Negative Binomial Parameterization with mean mu 
	and variance
	
		var = mu + alpha*mu^lambda
	
	The density function for a single data-point is then
	.. math::
		f(Y,r,p) = \frac{\gamma(Y + r)}{\gamma(Y + 1) \gamma(r)} (1 -p)^Y p^r
		f(Y, alpha, mu, lambda) = ...
		 \frac{\gamma(Y + alpha^-1)}{\gamma(Y + 1) \gamma(alpha^-1)} (\frac{\alpha \mu}{1 + \alpha \mu})^Y (1 + \alpha \mu)^(-alpha^-1)
	"""
	
	def __init__(self, tcr_frequencies, counts, num_reads):
		self.fmix = tcr_frequencies
		self.C = counts
		self.Nread = num_reads

	def fit(self, start_params = None,
				  stepsize = 1.0,
				  maxiter = 1000,
				  TOL = 1.0e-8,
				  method = "Newton",
				  show_convergence = False):

		if start_params is None:
			start_params = self.get_default_initparams(show_convergence)

		llhp = lambda p: self.llh(p[0], p[1], p[2])
		scorep = lambda p: self.score(p[0], p[1], p[2])
		llh_hess = nd.Jacobian(scorep, 1.0e-8)

		if method == "Newton":
			fitter = NewtonFitter(llhp,
								  scorep,
								  llh_hess)
			
			fittingresult = fitter.fit(start_params = start_params,
									   stepsize = stepsize,
									   TOL = TOL,
									   maxiter = maxiter,
									   show_convergence = show_convergence)
			pread, alpha, lmbda =fittingresult.params
			
		elif method == "SLSQP":
			opt_res = minimize(llhp,
							   start_params,
							   jac = scorep, 
				               method = "SLSQP",
				               bounds = [[0, None], [0, None], [0, None]],
				               options = {"disp":show_convergence})
			pread, alpha, lmbda = opt_res.x

		return NBVarTCRCountModel(pread, alpha, lmbda)

	def get_default_initparams(self, show_convergence):

		modelcalib = NB2Calibrator(self.fmix, self.C, self.Nread)
		nb2fit = modelcalib.fit()
		alpha0, pread0 = nb2fit.alpha, nb2fit.pread

		#Get lambda0 by a 1-d optimization
		llh_lmbda = partial(self.llh, pread0, alpha0)
		opt_res = minimize_scalar(lambda x: -1*llh_lmbda(x), bracket = [0,2])
		
		lmbda0 = opt_res.x
		return np.array([pread0, alpha0, lmbda0])

	def llh(self, pread, alpha, lmbda):
		mu = pread*self.fmix*self.Nread
		r,p = self.negbin_rp(mu, alpha, lmbda)

		llh = gammaln(self.C + r) - gammaln(r) - gammaln(self.C +1)
		llh += self.C*np.log(1 - p) + r*np.log(p)
		return np.sum(llh)

	def score(self, pread, alpha, lmbda):
		mu = pread*self.fmix*self.Nread

		alphainv = 1.0/alpha
		r,p = self.negbin_rp(mu, alpha, lmbda)

		dr_da = -1*mu**(2 - lmbda)*alphainv**2
		dp_da = -mu**(lmbda -1)/((1 + alpha*mu**(lmbda -1))**2)

		dr_dmu = ((2 - lmbda)*mu**(1 - lmbda))/alpha
		dp_dmu = -1 *(alpha*(lmbda - 1)*mu**(lmbda - 2))/(1 + alpha*mu**(lmbda - 1))**2

		dr_dlmbda = (-1*np.log(mu)*mu**(2 - lmbda))/alpha
		dp_dlmbda = (-alpha*np.log(mu)*mu**(lmbda -1))/(1 + alpha*mu**(lmbda -1))**2
		A1 = (digamma(self.C + r)- digamma(r) + np.log(p))
		A2 = (r/p - self.C/(1-p))

		dllh_da = (dr_da*A1 + dp_da*A2).sum()
		dllh_dpread = (self.fmix*self.Nread*(dr_dmu*A1 + dp_dmu*A2)).sum()
		dllh_dlmbda = (dr_dlmbda*A1 + dp_dlmbda*A2).sum()

		return np.array([dllh_dpread, dllh_da, dllh_dlmbda])

	@staticmethod
	def negbin_rp(mu, alpha, lmbda):
		r = mu**(2 - lmbda)/alpha
		amul = alpha*mu**(lmbda -1)		
		p = 1/(1 + amul)
		return r, p

class NBVarTCRCountModel:
	"Parameterized Negative Binomial 2 model"
	def __init__(self, pread, alpha, lmbda):
		self.pread = pread
		self.alpha = alpha
		self.lmbda = lmbda

	def predict_mean(self, tcr_frequencies, num_reads):
		return tcr_frequencies*self.pread*num_reads

	def predict_variance(self, tcr_frequencies, num_reads):
		mu = self.predict_mean(tcr_frequencies, num_reads)
		return mu + self.alpha*mu**self.lmbda

	def pmf(self, mu, count = 0):
		r,p = NBVarCalibrator.negbin_rp(mu, self.alpha, self.lmbda)
		return stats.nbinom.pmf(count, r, p)

	def predict_detection_probability(self, tcr_frequencies = 1.0,
											num_reads = 1,
											detect_thresh = 1):
		"""
		Models detection probability with negative binomial models assuming RNA receptor frequencies
		detect_thresh = Minimum number of reads before a TCR is considered "detected".
		"""
		#TODO: Implement a detection probability threshold by summing over the first argument of the pmf.
		mu = self.predict_mean(tcr_frequencies, num_reads)
		p_belowthresh = self.pmf(mu, count = np.arange(detect_thresh)[:,np.newaxis]).sum(axis =0)
		return 1.0 - p_belowthresh

	def get_prediction_interval(self, tcr_frequencies, num_reads, interval_size = 0.95):
		mu = self.predict_mean(tcr_frequencies, num_reads)
		r,p = NBVarCalibrator.negbin_rp(mu, self.alpha, self.lmbda)
		return stats.nbinom.interval(interval_size, r, p)