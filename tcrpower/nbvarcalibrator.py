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
		
		eta : variance scaling parameter
		lambda : variance exponential parameter

	Negative Binomial Parameterization with mean mu 
	and variance
	
		var = mu + eta*mu^lambda
	
	The density function for a single data-point is then
	.. math::
		f(Y,r,p) = \frac{\gamma(Y + r)}{\gamma(Y + 1) \gamma(r)} (1 -p)^Y p^r
		f(Y, eta, mu, lambda) = ...
		 \frac{\gamma(Y + eta^-1)}{\gamma(Y + 1) \gamma(eta^-1)} (\frac{\eta \mu}{1 + \eta \mu})^Y (1 + \eta \mu)^(-eta^-1)
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
			read_eff, eta, lmbda = fittingresult.params
			
		elif method == "SLSQP":
			opt_res = minimize(llhp,
							   start_params,
							   jac = scorep, 
				               method = "SLSQP",
				               bounds = [[0, None], [0, None], [0, None]],
				               options = {"disp":show_convergence})
			read_eff, eta, lmbda = opt_res.x

		return NBVarTCRCountModel(read_eff, eta, lmbda)

	def get_default_initparams(self, show_convergence):

		modelcalib = NB2Calibrator(self.fmix, self.C, self.Nread)
		nb2fit = modelcalib.fit()
		eta0, read_eff0 = nb2fit.eta, nb2fit.read_eff

		#Get lambda0 by a 1-d optimization
		llh_lmbda = partial(self.llh, read_eff0, eta0)
		opt_res = minimize_scalar(lambda x: -1*llh_lmbda(x), bracket = [0,2])
		
		lmbda0 = opt_res.x
		return np.array([read_eff0, eta0, lmbda0])

	def llh(self, read_eff, eta, lmbda):
		mu = read_eff*self.fmix*self.Nread
		r,p = self.negbin_rp(mu, eta, lmbda)

		llh = gammaln(self.C + r) - gammaln(r) - gammaln(self.C +1)
		llh += self.C*np.log(1 - p) + r*np.log(p)
		return np.sum(llh)

	def score(self, read_eff, eta, lmbda):
		mu = read_eff*self.fmix*self.Nread

		etainv = 1.0/eta
		r,p = self.negbin_rp(mu, eta, lmbda)

		dr_da = -1*mu**(2 - lmbda)*etainv**2
		dp_da = -mu**(lmbda -1)/((1 + eta*mu**(lmbda -1))**2)

		dr_dmu = ((2 - lmbda)*mu**(1 - lmbda))/eta
		dp_dmu = -1 *(eta*(lmbda - 1)*mu**(lmbda - 2))/(1 + eta*mu**(lmbda - 1))**2

		dr_dlmbda = (-1*np.log(mu)*mu**(2 - lmbda))/eta
		dp_dlmbda = (-eta*np.log(mu)*mu**(lmbda -1))/(1 + eta*mu**(lmbda -1))**2
		A1 = (digamma(self.C + r)- digamma(r) + np.log(p))
		A2 = (r/p - self.C/(1-p))

		dllh_da = (dr_da*A1 + dp_da*A2).sum()
		dllh_dread_eff = (self.fmix*self.Nread*(dr_dmu*A1 + dp_dmu*A2)).sum()
		dllh_dlmbda = (dr_dlmbda*A1 + dp_dlmbda*A2).sum()

		return np.array([dllh_dread_eff, dllh_da, dllh_dlmbda])

	@staticmethod
	def negbin_rp(mu, eta, lmbda):
		r = mu**(2 - lmbda)/eta
		amul = eta*mu**(lmbda -1)		
		p = 1/(1 + amul)
		return r, p

class NBVarTCRCountModel:
	"Parameterized Negative Binomial read count model"
	def __init__(self, read_eff, eta, lmbda):
		self.read_eff = read_eff
		self.eta = eta
		self.lmbda = lmbda

	def predict_mean(self, tcr_frequencies, num_reads):
		return tcr_frequencies*self.read_eff*num_reads

	def predict_variance(self, tcr_frequencies, num_reads):
		mu = self.predict_mean(tcr_frequencies, num_reads)
		return mu + self.eta*mu**self.lmbda

	def pmf(self, mu, count = 0):
		r,p = NBVarCalibrator.negbin_rp(mu, self.eta, self.lmbda)
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
		r,p = NBVarCalibrator.negbin_rp(mu, self.eta, self.lmbda)
		return stats.nbinom.interval(interval_size, r, p)

	def to_csv(self, outputpath):
		modelcoefs = pd.DataFrame([[self.read_eff, self.eta, self.lmbda]], 
			                       columns = ["read_efficiency", "eta", "lambda"])
		modelcoefs.to_csv(outputpath, index = False)

	def __str__(self):
		pad = 20
		re_str = "\n\tRead Efficiency".ljust(pad) + "= {}".format(self.read_eff)
		eta_str = "\n\tEta".ljust(pad) + "= {}".format(self.eta)
		lmbda_str = "\n\tLambda".ljust(pad) + "= {}".format(self.lmbda)
		return "Fitted Negative Binomial Model" + re_str + eta_str + lmbda_str

