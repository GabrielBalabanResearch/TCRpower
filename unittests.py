import numpy as np
from tcrpower.nb2calibrator import *
from tcrpower.nbvarcalibrator import NBVarCalibrator
from tcrpower.powercalc import TCRPowerCalculator
from scipy import stats
import numdifftools as nd

def get_testdata(intercept = True,
				 eta = 0.05,
				 Nread = 100000,
				 logrange = 6,
				 TCR_perlog = 15,
				 read_eff = 0.7,
				 lmbda = 2,
				 RANDSTATE = 42):

	logvals = np.repeat(10**np.arange(1,logrange), TCR_perlog)

	fmix = logvals/logvals.sum()

	mu = fmix*read_eff*Nread
	
	#Generate Y From a Negbin2 model
	r, p = NBVarCalibrator.negbin_rp(mu, eta, lmbda)
	C = stats.nbinom.rvs(r,
						 p,
						 size = len(fmix),
						 random_state = RANDSTATE)
	return C, fmix

def get_default_testparams(Nread = 10000, read_eff = 0.7, eta = 0.05, lmbda = 2.0):
	return Nread, read_eff, eta, lmbda

def test_parameterization_consistent():
	Nread, read_eff, eta, lmbda = get_default_testparams()

	C, fmix = get_testdata(Nread = Nread, 
						   read_eff = read_eff,
						   eta = eta)
	
	mu = Nread*read_eff*fmix
	
	r,p = rp_negbin_params(eta, mu)
	eta2, mu2 = eta_mu_negbin_params(r, p)
	assert eta == eta2
	assert np.linalg.norm(mu - mu2) < 1.0e-12
	print("Parameterization internal consistency test passed")

def test_PCCalibrator_llh():
	Nread, read_eff, eta, lmbda = get_default_testparams()

	C, fmix = get_testdata(Nread = Nread, 
						   read_eff = read_eff,
						   eta = eta,
						   lmbda= lmbda)

	mu = Nread*read_eff*fmix

	nb2_calibrator = NB2Calibrator(fmix, C, Nread)

	r,p = rp_negbin_params(eta, mu)
	logp_scipy = np.log(stats.nbinom.pmf(C, r, p)).sum()
	logp_model = nb2_calibrator.llh(read_eff, eta)
	
	assert np.abs(logp_model - logp_scipy) < 0.1
	print("LLH test passed")

def test_PCCalibrator_fdtest_paramderiv(show_results = False):
	#Tests that the score function matches a numerical derivative

	Nread, read_eff, eta, lmbda = get_default_testparams()
	C, fmix = get_testdata(Nread = Nread, 
						   read_eff = read_eff,
						   eta = eta,
						   lmbda = lmbda)

	nb2_calibrator = NB2Calibrator(fmix, C, Nread)
	x_true = np.array([read_eff, eta])

	x0 = x_true*0.5

	fdtester = FDTester(eps = 1.0e-8)
	grad_result = fdtester.test(x0,
							   	lambda x : nb2_calibrator.llh(x[0], x[1]),
							   	lambda x : nb2_calibrator.score(x[0], x[1]))
	if show_results:
		grad_result.report()
	
	assert grad_result.matches(relTOL = 1.0e-7)
	print("Finite difference test of score function passed")

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

def test_PCCalibrator_fit(show_results = False):
	Nread, read_eff, eta, lmbda = get_default_testparams(Nread = 1000000)
	C, fmix = get_testdata(eta = eta,
						   Nread = Nread,
						   read_eff = read_eff,
						   lmbda = lmbda,
						   TCR_perlog = 50)

	modelcalib = NB2Calibrator(fmix, C, Nread)
	
	pc_model = modelcalib.fit(show_convergence = show_results)
	assert np.abs(pc_model.read_eff - read_eff) < 0.2 
	assert np.abs(pc_model.eta - eta) < 0.002
	print("Parameter fitting test passed")

def test_TCRPowerCalculator_limit_of_detection_tcrfreq():
	Nread, read_eff, eta, lmbda = get_default_testparams(Nread = 1000000)
	C, fmix = get_testdata(eta = eta,
						   Nread = Nread,
						   read_eff = read_eff,
						   lmbda = lmbda,
						   TCR_perlog = 50)

	conf_level = 0.95

	modelcalib = NB2Calibrator(fmix, C, Nread)
	powercalc = TCRPowerCalculator(modelcalib.fit())
	
	#The lowest frequency TCR clone that can be detected with 95% reliability

	f_lod95 = powercalc.get_limit_of_detection_tcrfreq(Nread, conf_level)
	mu_lod95 = f_lod95*read_eff*Nread 

	r, p = rp_negbin_params(eta, mu_lod95)

	p_detect = 1 - stats.nbinom.pmf(0, r, p)
	assert np.abs(p_detect - conf_level) < 0.004
	print("TCR frequency limit of detection test passed")

def test_TCRPowerCalculator_limit_of_detection_nreads():
	Nread, read_eff, eta, lmbda = get_default_testparams(Nread = 1000000)
	C, fmix = get_testdata(eta = eta,
						   Nread = Nread,
						   read_eff = read_eff,
						   lmbda = lmbda,
						   TCR_perlog = 50)

	conf_level = 0.95

	modelcalib = NB2Calibrator(fmix, C, Nread)
	powercalc = TCRPowerCalculator(modelcalib.fit())
	
	test_tcr_freq = np.median(fmix)

	#The lowest frequency TCR clone that can be detected with 95% reliability
	nread_lod95 = powercalc.get_limit_of_detection_nreads(test_tcr_freq, conf_level)
	
	mu_lod95 = test_tcr_freq*read_eff*nread_lod95 

	r, p = rp_negbin_params(eta, mu_lod95)

	p_detect = 1 - stats.nbinom.pmf(0, r, p)
	assert np.abs(p_detect - conf_level) < 0.004
	print("Num reads limit of detection test passed")

def test_NBVarCalibrator_fit(show_results = False):
	Nread, read_eff, eta, lmbda = get_default_testparams(Nread = 1000000,
														eta = 0.25,
														lmbda = 1.5)
	C, fmix = get_testdata(eta = eta,
						   Nread = Nread,
						   read_eff = read_eff,
						   lmbda = lmbda,
						   TCR_perlog = 50)
	#print(C)
	true_params = np.array([read_eff, eta, lmbda])

	modelcalib = NBVarCalibrator(fmix, C, Nread)
	
	pc_model = modelcalib.fit(show_convergence = show_results,
		                      stepsize = 1.0,
		                      method = "Newton")

	if show_results:
		print("Estimated", np.array([pc_model.read_eff, pc_model.eta, pc_model.lmbda]).round(3))
		print("True", true_params)
	assert np.abs(pc_model.read_eff - read_eff) < 0.2 
	assert np.abs(pc_model.eta - eta) < 0.03
	print("PCVAR Parameter fitting test passed")

def test_TCRPowerCalculator_NBvar_limit_of_detection_tcrfreq():
	Nread, read_eff, eta, lmbda = get_default_testparams(Nread = 1000000,
		                                                eta = 0.25,
		                                                lmbda = 1.5)
	C, fmix = get_testdata(eta = eta,
						   Nread = Nread,
						   read_eff = read_eff,
						   lmbda = lmbda,
						   TCR_perlog = 50)
	conf_level = 0.95

	modelcalib = NBVarCalibrator(fmix, C, Nread)
	powercalc = TCRPowerCalculator(modelcalib.fit())
	
	#The lowest frequency TCR clone that can be detected with 95% reliability

	f_lod95 = powercalc.get_limit_of_detection_tcrfreq(Nread, conf_level)
	mu_lod95 = f_lod95*read_eff*Nread 

	r, p = NBVarCalibrator.negbin_rp(mu_lod95, eta, lmbda)

	p_detect = 1 - stats.nbinom.pmf(0, r, p)
	assert np.abs(p_detect - conf_level) < 0.004
	print("NBVar TCR frequency limit of detection test passed")

def test_TCRPowerCalculator_NBVar_limit_of_detection_nreads():
	Nread, read_eff, eta, lmbda = get_default_testparams(Nread = 1000000,
														eta = 0.25,
		                                                lmbda = 1.5)
	C, fmix = get_testdata(eta = eta,
						   Nread = Nread,
						   read_eff = read_eff,
						   lmbda = lmbda,
						   TCR_perlog = 50)

	conf_level = 0.95

	modelcalib = NBVarCalibrator(fmix, C, Nread)
	powercalc = TCRPowerCalculator(modelcalib.fit())
	
	test_tcr_freq = np.median(fmix)

	#The lowest frequency TCR clone that can be detected with 95% reliability
	nread_lod95 = powercalc.get_limit_of_detection_nreads(test_tcr_freq, conf_level)
	mu_lod95 = test_tcr_freq*read_eff*nread_lod95 

	r, p = NBVarCalibrator.negbin_rp(mu_lod95, eta, lmbda)

	p_detect = 1 - stats.nbinom.pmf(0, r, p)
	assert np.abs(p_detect - conf_level) < 0.004
	print("NBVar Num reads limit of detection test passed")

if __name__ == "__main__":
	test_parameterization_consistent()
	test_PCCalibrator_llh()
	test_PCCalibrator_fdtest_paramderiv(show_results = False)
	test_PCCalibrator_fit(show_results = False)
	test_TCRPowerCalculator_limit_of_detection_tcrfreq()
	test_TCRPowerCalculator_limit_of_detection_nreads()
	test_NBVarCalibrator_fit(show_results = False)
	test_TCRPowerCalculator_NBvar_limit_of_detection_tcrfreq()
	test_TCRPowerCalculator_NBVar_limit_of_detection_nreads()