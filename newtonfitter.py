import numpy as np

class NewtonFitter():
	def __init__(self, llh, score, hess, output_layer = 0):
		self.llh = llh
		self.score = score
		self.hess = hess
		self.output_layer = output_layer

	def fit(self, 
			start_params,
			stepsize = 1.0,
			TOL = 1.0e-8,
			maxiter = 1000,
			show_convergence = False):

		#Initialization
		params_next = start_params
		self.residuals = []

		if show_convergence: 
			convergence_reporter = ConvergenceReporter("Fitting Parameters with Newton", 
														maxiter,
														self.output_layer)

		error = np.linalg.norm(self.score(params_next))

		i = 0
		while error > TOL:
			i += 1
			if i > maxiter:
				raise Exception("Number of Newton Iterations Exceeded {0}".format(maxiter))

			params_prev = params_next
			
			b = self.score(params_next)

			hess = self.hess(params_next)

			d_params = np.linalg.solve(hess, -1*b)
			error = np.linalg.norm(b)
			self.residuals.append(error)
			
			params_next = params_prev + d_params*stepsize

			if np.isnan(params_next).any():
				raise Exception("Newton failed, NAN in parameters.")

			if show_convergence: 
				convergence_reporter.reportiter(i,
												error, 
												self.llh(params_next),
												params_next)
		return FittingResult(params_next,
							 i,
							 TOL)

class FittingResult(object):
	def __init__(self, params, iterations, TOL):
		self.params = params
		self.iterations = iterations
		self.TOL = TOL

class ConvergenceReporter(object):
	def __init__(self, inimessage, maxiter, output_layer = 0):
		self.maxiter = maxiter
		self.iter_digits = int(np.ceil(np.log10(self.maxiter))) +1
		self.output_pad = "\t"*output_layer

		print(self.output_pad + inimessage)
		title = self.output_pad + "itr".ljust(self.iter_digits) + \
				"error".ljust(11) +  "llh".ljust(11) + "params"
		print(title)
		print(self.output_pad + "-"*(self.iter_digits + 34))

	def reportiter(self, i, error, llh, params):
		msg = str(i).ljust(self.iter_digits) +\
		  "{:.3g}".format(error).ljust(11) +\
		  "{:.3g}".format(llh).ljust(11) +\
		  " ".join(["{:.3g}".format(p) for p in params])
		print(self.output_pad + msg)