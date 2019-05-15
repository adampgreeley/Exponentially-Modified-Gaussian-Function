# Exponentially-Modified Gaussian functions for Python
# April 30, 2019
#
#

import numpy as np
from scipy.special import erf
from scipy.special import erfc



#-- PURPOSE: compute the cumulative probability for a normalized Gaussian
def pnf(x):
	#-- x is scalar or array
	#-- function returns probability with same shape as x
	a = x < 0
	b = x >= 0
	p = x
	m_sqrt2 = np.sqrt(2)
	p[b] = ( (1 + erf(x[b] / m_sqrt2)) / 2)
	p[a] = ( (erfc(-x[a] / m_sqrt2)) / 2)
	return p



#-- PURPOSE: compute exponentially-modified Gaussian PDF
def exgausspdf(x,mu,sig,tau):

	#-- Given ex-Gaussian parameters mu, sig, tau, return density at x. 
	#-- mu, sig, tau are scalar; x is either scalar or array.
	#-- function returns density with same shape as x
	arg1 = (mu / tau) + ((sig**2) / (2 * (tau**2)) ) - (x / tau)
	arg2 = ((x - mu) - ((sig**2) / tau)) / sig
	f = (1 / tau) * np.exp(arg1) * pnf(arg2)
	return f


