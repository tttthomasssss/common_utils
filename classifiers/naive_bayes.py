from __future__ import division
__author__ = 'thk22'

import math

from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np


class NaiveBayesSmoothing(object):
	@staticmethod
	def lidstone_smoothing(fcc, alpha=1.):
		fcc += alpha
		ncc = fcc.sum(axis=1).reshape(-1, 1)
		return (fcc / ncc), (np.log(fcc) - np.log(ncc))

	@staticmethod
	def lidstone_smoothing_no_renormalisation(fcc, alpha=1.): # <-- Similar to the prior_smoothing but it is uniform
		ncc = fcc.sum(axis=1).reshape(-1, 1)
		fcc += (alpha / fcc.shape[0])

		return (fcc / ncc), (np.log(fcc) - np.log(ncc))

	@staticmethod
	def jelinek_mercer_smoothing(fcc, lambada=0.5, p_w=None):
		p_w_given_c = (fcc / fcc.sum(axis=1).reshape(-1, 1)) * (1 - lambada)
		p_w = ((fcc.sum(axis=0) / fcc.sum()).reshape(1, -1) * lambada) if p_w is None else p_w * lambada

		return (p_w + p_w_given_c), np.nan_to_num(np.log(p_w + p_w_given_c))

	@staticmethod
	def prior_smoothing(fcc, mu=0.95): # <-- I have no idea why this works, but it works better than anything else
		p_c = (fcc.sum(axis=1).reshape(-1, 1) / fcc.sum())

		enum = (fcc + (mu * p_c))
		denom = (fcc.sum(axis=1).reshape(-1, 1))
		p_w_given_c = enum / denom

		return p_w_given_c, np.log(enum) - np.log(denom)

	@staticmethod
	def dirichlet_smoothing(fcc, mu=0.95, p_w=None):
		#p_w = (fcc.sum(axis=1).reshape(-1, 1) / fcc.sum())
		p_w = (fcc.sum(axis=0) / fcc.sum()).reshape(1, -1) if p_w is None else p_w
		p_w_given_c = (fcc + (mu * p_w)) / ((fcc + mu).sum(axis=1).reshape(-1, 1))
		log_p_w_given_c = np.log(fcc + (mu * p_w)) - np.log((fcc + mu).sum(axis=1).reshape(-1, 1))

		return p_w_given_c, log_p_w_given_c

	@staticmethod
	def absolute_discounting(fcc, sigma=0.6, p_w=None):
		p_w = (fcc.sum(axis=0) / fcc.sum()).reshape(1, -1) if p_w is None else p_w

		num_unique_w = np.zeros((fcc.shape[0], 1))
		for row_idx in np.arange(fcc.shape[0]):
			num_unique_w[row_idx] = np.count_nonzero(fcc[row_idx])

		p_w_given_c = (np.maximum(fcc - sigma, 0.) + (np.dot((sigma * np.asmatrix(num_unique_w)), p_w))) / (fcc.sum(axis=1).reshape(-1, 1))

		return p_w_given_c, np.nan_to_num(np.log(p_w_given_c))

	@staticmethod
	def two_stage_smoothing(fcc, lambada=0.6, mu=100, p_w=None):
		p_w = (fcc.sum(axis=0) / fcc.sum()).reshape(1, -1) if p_w is None else p_w

		dirichlet_stage, _ = NaiveBayesSmoothing.dirichlet_smoothing(fcc, mu, p_w)

		p_w_given_c = ((1 - lambada) * dirichlet_stage) + (lambada * p_w)

		return p_w_given_c, np.nan_to_num(np.log(p_w_given_c))

	@staticmethod
	def calc_lidstone_damping_factor_tokens(vocab_size, n_tokens, alpha=1.):
		damping_magnitude_tokens = math.floor(math.log10(vocab_size * alpha)) / math.floor(math.log10(n_tokens))
		damping_factor_tokens = 10 ** damping_magnitude_tokens

		return damping_factor_tokens

	@staticmethod
	def calc_lidstone_damping_factor_vocab(vocab_size, sample_vocab_size, alpha=1.):
		damping_magnitude_vocab = math.floor(math.log10(vocab_size * alpha)) / math.floor(math.log10(sample_vocab_size))
		damping_factor_vocab = 10 ** damping_magnitude_vocab

		return damping_factor_vocab


class NaiveBayesClassifier(object):

	def __init__(self, smoothing_fn=NaiveBayesSmoothing.lidstone_smoothing, **kwargs):
		self.classes_ = None
		self.priors_ = None
		self.log_priors_ = None
		self.feature_counts_ = None
		self.probs_ = None
		self.log_probs_ = None
		self.smoothing_fn_ = smoothing_fn
		self.smoothing_args_ = kwargs.pop('smoothing_args', 1.)

	def fit(self, X, y, fit_priors=True):
		X = self._to_csr(X)

		self._fit_priors(y, fit_priors)

		self._fit_features(X, y)

	def predict(self, X):
		jll = self._joint_log_likelihood(X)

		return self.classes_[np.argmax(jll, axis=1)]

	def predict_proba(self, X):
		return self._joint_log_likelihood(X)

	def _to_csr(self, X):
		if (sparse.isspmatrix_csr(X)):
			return X
		elif (sparse.issparse(X)):
			return X.tocsr()
		else: # i.e. numpy arrays/ndmatrices
			return sparse.csr_matrix(X)

	# Margin of Confidence
	def _estimate_log_prob_evidence(self, ranked_label_idx):
		# Evidence sets for most-likely class and second most-likely class; (the un-indexed equations in section D. of Sharma & Bilgic (2013))
		mlc_log_evidence_delta = self.log_probs_[ranked_label_idx[:, 0]] - self.log_probs_[ranked_label_idx[:, 1]]
		smlc_log_evidence_delta = self.log_probs_[ranked_label_idx[:, 1]] - self.log_probs_[ranked_label_idx[:, 0]]

		mlc_log_evidence_idx = np.where(mlc_log_evidence_delta > 0.)
		smlc_log_evidence_idx = np.where(smlc_log_evidence_delta > 0.)

		# Actual per-instance evidence (equations 17 & 18 in Sharma & Bilgic (2013))
		mlc_log_evidence = np.zeros(mlc_log_evidence_delta.shape)
		mlc_log_evidence[mlc_log_evidence_idx] = mlc_log_evidence_delta[mlc_log_evidence_idx]
		smlc_log_evidence = np.zeros(smlc_log_evidence_delta.shape)
		smlc_log_evidence[smlc_log_evidence_idx] = smlc_log_evidence_delta[smlc_log_evidence_idx]

		E = mlc_log_evidence.sum(axis=1) + smlc_log_evidence.sum(axis=1)

		return E

	# Margin of Confidence
	def _estimate_prob_evidence(self, ranked_label_idx):
		# Evidence sets for most-likely class and second most-likely class; (the un-indexed equations in section D. of Sharma & Bilgic (2013))
		mlc_evidence_delta = self.probs_[ranked_label_idx[:, 0]] / self.probs_[ranked_label_idx[:, 1]]
		smlc_evidence_delta = self.probs_[ranked_label_idx[:, 1]] / self.probs_[ranked_label_idx[:, 0]]

		mlc_evidence_idx = np.where(mlc_evidence_delta > 1.)
		smlc_evidence_idx = np.where(smlc_evidence_delta > 1.)

		# Actual per-instance evidence (equations 17 & 18 in Sharma & Bilgic (2013))
		mlc_evidence = np.zeros(mlc_evidence_delta.shape)
		mlc_evidence[mlc_evidence_idx] = mlc_evidence_delta[mlc_evidence_idx]
		smlc_evidence = np.zeros(smlc_evidence_delta.shape)
		smlc_evidence[smlc_evidence_idx] = smlc_evidence_delta[smlc_evidence_idx]

		E = mlc_evidence.prod(axis=1) * smlc_evidence.prod(axis=1)

		return E

	def estimate_evidence(self, ranked_label_idx, use_log_probs=True):

		E = self._estimate_log_prob_evidence(ranked_label_idx) if use_log_probs else self._estimate_prob_evidence(ranked_label_idx)

		'''
		# Evidence for 1 instance at a time
		EE = np.zeros((ranked_label_idx.shape[0], 2))
		for idx, curr_label_ranking in enumerate(ranked_label_idx):
			mlc_log_evidence_idx = np.where((self.log_probs_[curr_label_ranking[0]] - self.log_probs_[curr_label_ranking[1]]) > 0.)[0]
			smlc_log_evidence_idx = np.where((self.log_probs_[curr_label_ranking[1]] - self.log_probs_[curr_label_ranking[0]]) > 0.)[0]

			mlc_log_evidence = (self.log_probs_[curr_label_ranking[0]][mlc_log_evidence_idx] - self.log_probs_[curr_label_ranking[1]][mlc_log_evidence_idx]).sum()
			smlc_log_evidence = (self.log_probs_[curr_label_ranking[1]][smlc_log_evidence_idx] - self.log_probs_[curr_label_ranking[0]][smlc_log_evidence_idx]).sum()

			EE[idx, 0] = mlc_log_evidence
			EE[idx, 1] = smlc_log_evidence

			print 'MLC EVIDENCE=%.2f; SMLC EVIDENCE=%.2f' % (mlc_log_evidence, smlc_log_evidence)
		'''

		return E

	def _fit_priors(self, y, fit_priors):
		self.priors_ = np.bincount(y) / y.sum() if (fit_priors) else np.ones(np.unique(y).shape) * (1 / np.unique(y).shape[0])
		self.classes_ = np.arange(self.priors_.shape[0])
		self.log_priors_ = np.nan_to_num(np.log(self.priors_))

	def _fit_features(self, X, y):
		self.feature_counts_ = np.zeros((self.priors_.shape[0], X.shape[1]))

		# Count Features
		for y_i in self.classes_:
			self.feature_counts_[y_i, :] = X[np.where(y == y_i)].sum(axis=0)

		self.probs_, self.log_probs_ = self.smoothing_fn_(self.feature_counts_, *self.smoothing_args_ )

	def _joint_log_likelihood(self, X):
		return safe_sparse_dot(X, self.log_probs_.T) + self.log_priors_