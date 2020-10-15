import numpy as np 
import pandas as pd
from scipy.special import gamma, gammaln, digamma, polygamma

class ModelCalibrator(object):
	"""
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

		

	Negative Binomial Parameterization is the NB2 model with mean mu 
	and variance
	
		var = mu + alpha*mu^2
	
	The density function for a single data-point is then
	.. math::
		f(Y,r,p) = \frac{\gamma(Y + r)}{\gamma(Y + 1) \gamma(r)} (1 -p)^Y p^r
		f(Y, alpha, mu) = \frac{\gamma(Y + alpha^-1)}{\gamma(Y + 1) \gamma(alpha^-1)} (\frac{\alpha \mu}{1 + \alpha \mu})^Y (1 + \alpha \mu)^-r
	"""
	
	def __init__(self, fmix, C, Nread):
		self.fmix = fmix
		self.C = C
		self.Nread = Nread

	def fit(self):
		pass
		#Do negbin calculations

	def llh(self, pread, alpha):
		mu = self.fmix*pread*self.Nread

		alphainv = alpha**-1
		llh = gammaln(self.C + alphainv) - gammaln(alphainv) - gammaln(self.C +1)
		llh += self.C*np.log(alpha*mu) - (self.C + alphainv)*np.log(1 + alpha*mu)
		return np.sum(llh)

	#First derivative Functions
	def score(self, pread, alpha):
		mu = self.fmix*pread*self.Nread
		
		#Alpha derivative
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