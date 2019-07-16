"""
Exponentially-Modified Gaussian functions for Python

Used  in
*Characterizing the System Impulse Response Function From Photon-Counting LiDAR Data*
by Adam P. Greeley , Thomas A. Neumann, Nathan T. Kurtz, Thorsten Markus, and Anthony J. Martino
to be published in **IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING**.
Digital Object Identifier 10.1109/TGRS.2019.2907230
See https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution for
a description of the Exponentially-Modified Gaussian function.
See <https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution> for
a description of the Exponentially-Modified Gaussian function.
The regular entry point is **exgausspdf()**.
"""


import numpy as np
from scipy.optimize import fmin
from scipy.special import erf
from scipy.special import erfc



def pnf(x):
	"""Helper function to compute the cumulative probability for a normalized Gaussian
	
	:param x: x is scalar or array. Note that x is overwritten during the function evaluation.
	:return: probability with same shape as x
	
	The pnf is defined as the probability that x<X when X is Gaussian distributed.
	Note that for numerical stability, a different formula is used for
	positive and negative arguments.
	"""
	
	a = x < 0
	b = x >= 0
	p = x
	m_sqrt2 = np.sqrt(2)
	p[b] = ( (1 + erf(x[b] / m_sqrt2)) / 2)
	p[a] = ( (erfc(-x[a] / m_sqrt2)) / 2)
	return p



def exgausspdf(x,mu,sig,tau):
	"""Compute exponentially-modified Gaussian PDF with parameters mu, sig, and tau.
	
	:param x: x is scalar or array of values at which to compute the density.
	:param mu: scalar: central tendency.
	:param sig: scalar: symmetric variablity.
	:param tau: scalar: Exponential decay.
	:return: probability with same shape as x.
	
	"""
	
	arg1 = (mu / tau) + ((sig**2) / (2 * (tau**2)) ) - (x / tau)
	arg2 = ((x - mu) - ((sig**2) / tau)) / sig
	f = (1 / tau) * np.exp(arg1) * pnf(arg2)
	return f



def emglike(params, xdata):
	"""Compute negative log sum of exponentially-modified Gaussian PDF with paramters mu, sig, and tau.
	
	:param params: 3-element EMG parameter scalar vector: element (1) mu, (2) sig, and (3) tau.
	:return: negative log sum
	
	"""
	
	# check for xdata row vector and convert to column vector
	if (xdata.ndim > 1) and (xdata.shape[0] < xdata.shape[1]):
		xdata = xdata.T
	
	# calculate the EMG probability distribution
	y = exgausspdf(xdata, params[0], params[1], params[2])
	
	# compute the negative log-sum for the EMG PDF
	logL = (-1) * np.sum(np.log(y))
	
	return logL



def emglikermse(params, xdata, ydata, y0_noise_floor):
	"""Compute the RMSE between exponentially-modified Gaussian and reference waveform.
	
	:param params: 3-element EMG parameter scalar vector: element (1) mu, (2) sig, and (3) tau.
	:param xdata: x is scalar or array of values at which to compute the EMG density and corresponding waveform.
	:param ydata: y is scalar or array of reference waveform values.
	:param y0_noise_floor: noise floor of reference waveform.
	:return: negative log sum
	
	"""
	
	# check for xdata row vector and convert to column vector
	if (xdata.ndim > 1) and (xdata.shape[0] < xdata.shape[1]):
		xdata = xdata.T
	
	# check for ydata row vector and convert to column vector
	if (ydata.ndim > 1) and (ydata.shape[0] < ydata.shape[1]):
		ydata = ydata.T
	
	# subtract noise floor from waveform for fitting
	y0 = ydata - y0_noise_floor
	
	# create EMG distribution from EMG parameters and x-data passed into this function
	y = exgausspdf(xdata, params[0], params[1], params[2])
	
	# calculate scaling factor for EMG distribution
	emg_scale = np.trapz(y0, xdata) / np.trapz(y, xdata)
	
	# scale EMG distribution to same area as ATM waveform
	y = y * emg_scale
	
	# calculate RMSE between EMG distribution and ATM pulse
	rmse = np.sqrt( np.sum( (y0 - y)**2) / y.size)
	
	return rmse



def fMinSearchStore(initial_params, xdata):
	"""Fit exponentially-modified Gaussain to ATLAS photon data using MLE.
	
	:param initial_params: initial guess for 3-element EMG parameters: element (1) mu, (2) sig, and (3) tau.
	:param xdata: x is scalar or array of values at which to compute the EMG density.
	
	"""
	
	fit_params = fmin(emglike, initial_params, args=(xdata,), retall=False, disp=False)
	
	return fit_params



def fMinSearchStore_EMG_RMSE(initial_params, xdata, ydata, y_noise_floor):
	"""Fit exponentially-modified Gaussain to ATM pulse using RMSE.
	
	:param initial_params: initial guess for 3-element EMG parameters: element (1) mu, (2) sig, and (3) tau.
	:param xdata: x is scalar or array of values at which to compute the EMG density and corresponding waveform.
	:param ydata: y is scalar or array of reference waveform values.
	:param y0_noise_floor: noise floor of reference waveform.
	
	"""
	
	fit_params = fmin(emglikermse, initial_params, args=(xdata, ydata, y_noise_floor), retall=False, disp=False)
	
	return fit_params




