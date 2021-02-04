import numpy as np 
import pandas as pd
import statsmodels.api as sm
import numdifftools as nd
import warnings
from scipy.special import gamma, gammaln, digamma, polygamma
from tcrpower.newtonfitter import NewtonFitter
from functools import partial

class PCVarPowerCalibrator(object):
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
				  show_convergence = False):

		if start_params is None:
			start_params = self.get_default_initparams()

		scorep = lambda p: self.score(p[0], p[1], p[2])
		llh_hess = nd.Jacobian(scorep, 1.0e-8)

		fitter = NewtonFitter(lambda p: self.llh(p[0], p[1], p[2]),
							  scorep,
							  llh_hess)
		
		fittingresult = fitter.fit(start_params = start_params,
								   stepsize = stepsize,
								   TOL = TOL,
								   maxiter = maxiter,
								   show_convergence = show_convergence)
		
		return PCModel(fittingresult.params[0], 
					   fittingresult.params[1],
					   fittingresult.params[2])

	def get_default_initparams(self):
		#Get beta parameters from Poisson model
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			poisson_fam = sm.families.Poisson(link = sm.genmod.families.links.identity())
			beta0 = sm.GLM(self.Y, self.X, poisson_fam).fit().params

		from IPython import embed; embed()
		#Set lmbda = 2 and fit alpha0



#		#Get alpha, beta from NB2 model
#		nb2_model = NB2RegressionFitter()
#		dim_X = self.X.shape[1]

#		xvars = ["X{}".format(i) for i in range(dim_X)]
#		XY_df= XY_df = pd.DataFrame(np.hstack([self.Y[:, np.newaxis], self.X]), 
#									columns = ["Y"] + xvars)

#		nb2_fit = nb2_model.fit("Y",
#								 xvars,
#					  			 XY_df,
#					  			 method = "Newton",
#					  			 maxiter = 50,
#					  			 TOL = 1.0e-7,
#				 	 			 show_convergence = False)

#		beta0 = nb2_fit.params_arr[:dim_X]
#		alpha0 = nb2_fit.params_arr[dim_X]

		#Get lambda by a 1-d optimization
		llh_lmbda = partial(self.llh, beta0, alpha0)
		opt_res = minimize_scalar(lambda x: -1*llh_lmbda(x), bracket = [0,2])
		
		lmbda0 = opt_res.x

		return np.hstack([beta0, alpha0, lmbda0])


	def llh(self, pread, alpha, lmbda):
		mu = self.fmix*pread*self.Nread
		r,p = self.negbin_rp(mu, alpha, lmbda)

		llh = gammaln(self.C + r) - gammaln(r) - gammaln(self.C +1)
		llh += self.C*np.log(1 - p) + r*np.log(p)
		return np.sum(llh)

	def score(self, pread, alpha, lmbda):
		mu = self.fmix*pread*self.Nread

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
		dllh_dbeta = np.dot(self.X.T, dr_dmu*A1 + dp_dmu*A2)
		dllh_dlmbda = (dr_dlmbda*A1 + dp_dlmbda*A2).sum()

		return np.hstack([dllh_dbeta, [dllh_da, dllh_dlmbda]])

	@staticmethod
	def negbin_rp(mu, alpha, lmbda):
		r = mu**(2 - lmbda)/alpha
		amul = alpha*mu**(lmbda -1)		
		p = 1/(1 + amul)
		return r, p