import numpy as np
from calibrate import *
from scipy import stats
import numdifftools as nd

def get_testdata(intercept = True,
				 alpha = 0.05,
				 Nread = 100000,
				 logrange = 6,
				 TCR_perlog = 15,
				 pread = 0.7,
				 RANDSTATE = 42):

	logvals = np.repeat(10**np.arange(1,logrange), TCR_perlog)

	fmix = logvals/logvals.sum()

	mu = fmix*pread*Nread
	
	#Generate Y From a Negbin2 model
	r, p = rp_negbin_params(alpha, mu)
	
	#scipy uses 1 - p = our p
	C = stats.nbinom.rvs(r,
						 p,
						 size = len(fmix),
						 random_state = RANDSTATE)
	return C, fmix

def get_default_testparams():
	Nread = 1000
	pread = 0.7
	alpha = 0.05
	return Nread, pread, alpha

def test_parameterization_consistent():
	Nread, pread, alpha = get_default_testparams()

	C, fmix = get_testdata(Nread = Nread, 
						   pread = pread,
						   alpha = alpha)
	
	mu = Nread*pread*fmix
	
	r,p = rp_negbin_params(alpha, mu)
	alpha2, mu2 = alpha_mu_negbin_params(r, p)
	assert alpha == alpha2
	assert np.linalg.norm(mu - mu2) < 1.0e-12
	print("Parameterization internal consistency test passed")

def test_ModelCalibrator_llh():
	Nread, pread, alpha = get_default_testparams()

	C, fmix = get_testdata(Nread = Nread, 
						   pread = pread,
						   alpha = alpha)

	mu = Nread*pread*fmix

	nb2_calibrator = ModelCalibrator(fmix, C, Nread)

	r,p = rp_negbin_params(alpha, mu)
	logp_scipy = np.log(stats.nbinom.pmf(C, r, p)).sum()
	logp_model = nb2_calibrator.llh(pread, alpha)
	
	assert np.abs(logp_model - logp_scipy) < 0.1
	print("LLH test passed")

def test_ModelCalibrator_fdtest_paramderiv(show_results = False):
	Nread, pread, alpha = get_default_testparams()
	C, fmix = get_testdata(Nread = Nread, 
						   pread = pread,
						   alpha = alpha)

	nb2_calibrator = ModelCalibrator(fmix, C, Nread)
	x_true = np.array([pread, alpha])

	x0 = x_true*0.5

	fdtester = FDTester(eps = 1.0e-8)
	grad_result = fdtester.test(x0,
							   	lambda x : nb2_calibrator.llh(x[0], x[1]),
							   	lambda x : nb2_calibrator.score(x[0], x[1]))
	if show_results:
		grad_result.report()
	
	assert grad_result.matches(relTOL = 1.0e-7)
	
#################################################
#Finite difference tester for the score function
#################################################
class FDTester(object):
	def __init__(self, eps = 1.0e-8):
		self.eps = eps

	def test(self,x, 
				  f,
				  df):

		df_fd = nd.Gradient(f, self.eps)(x)
		grad_result = FDTestResult(df_fd,
								   df(x))
		return grad_result
		
class FDTestResult(object):
	def __init__(self, df_fd, df_ana):
		self.df_fd = df_fd
		self.df_ana = df_ana

	def report(self):
		print("\nFD Gradient Test")
		print("Finite Difference")
		print(self.df_fd)
		print("Analytical")
		print(self.df_ana)

	def matches(self, relTOL = 1.0e-7):
		return np.linalg.norm(self.df_fd - self.df_ana)/np.linalg.norm(self.df_ana) < relTOL
#################################################

def test_ModelCalibrator_fit(show_results = False):
	Nread, pread, alpha = get_default_testparams()
	C, fmix = get_testdata(alpha = alpha,
						   Nread = Nread,
						   pread = pread)


	modelcalib = ModelCalibrator(fmix, C, Nread)
	
	from IPython import embed; embed()


if __name__ == "__main__":
	test_parameterization_consistent()
	test_ModelCalibrator_llh()
	test_ModelCalibrator_fdtest_paramderiv(show_results = True)
	test_ModelCalibrator_fit(show_results = True)