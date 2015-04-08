from __future__ import division
__author__ = 'thk22'

from classifiers.naive_bayes import NaiveBayesSmoothing


def lidstone_smoothing(fcc, alpha):
	return NaiveBayesSmoothing.lidstone_smoothing(fcc, alpha)


def lidstone_smoothing_no_renormalisation(fcc, alpha): # <-- Similar to the prior_smoothing but it is uniform
	return NaiveBayesSmoothing.lidstone_smoothing_no_renormalisation(fcc, alpha)


def jelinek_mercer_smoothing(fcc, lambada, p_w=None):
	return NaiveBayesSmoothing.jelinek_mercer_smoothing(fcc, lambada, p_w=p_w)


def prior_smoothing(fcc, mu): # <-- I have no idea why this works, but it works better than anything else
	return NaiveBayesSmoothing.prior_smoothing(fcc, mu)


def dirichlet_smoothing(fcc, mu, p_w=None):
	return NaiveBayesSmoothing.dirichlet_smoothing(fcc, mu, p_w=p_w)


def absolute_discounting(fcc, sigma, p_w=None):
	return NaiveBayesSmoothing.absolute_discounting(fcc, sigma, p_w=p_w)


def two_stage_smoothing(fcc, lambada, mu, p_w=None):
	return NaiveBayesSmoothing.two_stage_smoothing(fcc, lambada, mu, p_w=p_w)


def calc_lidstone_damping_factor_tokens(vocab_size, n_tokens, alpha):
	return NaiveBayesSmoothing.calc_lidstone_damping_factor_tokens(vocab_size, n_tokens, alpha)


def calc_lidstone_damping_factor_vocab(vocab_size, sample_vocab_size, alpha):
	return NaiveBayesSmoothing.calc_lidstone_damping_factor_vocab(vocab_size, sample_vocab_size, alpha)