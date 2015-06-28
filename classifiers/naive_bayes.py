from __future__ import division
__author__ = 'thk22'
import itertools
import math

from scipy import sparse
from scipy.optimize import newton
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.utils.extmath import logsumexp
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np

from common import wittgenstein

class NaiveBayesSmoothing(object): # TODO: Convert to Smoothing Mixin
	@staticmethod
	def lidstone_smoothing(fcc, alpha=1.):
		fcc += alpha
		ncc = fcc.sum(axis=1).reshape(-1, 1)

		return (fcc / ncc), (np.log(fcc) - np.log(ncc))

	@staticmethod
	def lidstone_smoothing_no_renormalisation(fcc, alpha=1.): # <-- Similar to the prior_smoothing but it is uniform
		try:
			ncc = fcc.sum(axis=1).reshape(-1, 1)
		except ValueError:
			ncc = fcc.sum()
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

		return p_w_given_c, np.nan_to_num(log_p_w_given_c)

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
		denom = np.nan_to_num(math.floor(np.log10(n_tokens)))

		# Avoid division by 0
		denom = denom if denom > 0. else 1.

		damping_magnitude_tokens = math.floor(np.log10(vocab_size * alpha)) / denom
		damping_factor_tokens = 10 ** damping_magnitude_tokens

		return damping_factor_tokens

	@staticmethod
	def calc_lidstone_damping_factor_vocab(vocab_size, sample_vocab_size, alpha=1.):
		damping_magnitude_vocab = math.floor(math.log10(vocab_size * alpha)) / math.floor(math.log10(sample_vocab_size))
		damping_factor_vocab = 10 ** damping_magnitude_vocab

		return damping_factor_vocab


class NaiveBayesSmoothingMixin(object):
	def lidstone_smoothing(self,
						   fcc, alpha=1.):
		if (sparse.issparse(fcc)):
			fcc, ncc = self._sparse_lidstone_smoothing(fcc, alpha)
		else:
			fcc += alpha
			try:
				ncc = fcc.sum(axis=1).reshape(-1, 1)
			except ValueError:
				ncc = fcc.sum()

		return (fcc / ncc), (np.log(fcc) - np.log(ncc))

	def _sparse_lidstone_smoothing(self, fcc, alpha):
		return fcc.sum(axis=0) + np.full(fcc.shape[1], alpha), fcc.sum() + (alpha * fcc.shape[1])

	def lidstone_smoothing_token_scaling(self, fcc, alpha=1.):
		damping_factor = self.calc_lidstone_damping_factor_tokens(vocab_size=fcc.shape[1], n_tokens=fcc.sum(), alpha=alpha)

		return self.lidstone_smoothing(fcc, (1 / damping_factor))

	def lidstone_smoothing_vocab_scaling(self, fcc, alpha=1.):
		damping_factor = self.calc_lidstone_damping_factor_vocab(vocab_size=fcc.shape[1], sample_vocab_size=np.count_nonzero(fcc.sum(axis=0)), alpha=alpha)

		return self.lidstone_smoothing(fcc, (1 / damping_factor))

	def lidstone_smoothing_no_renormalisation(self, fcc, alpha=1.): # <-- Similar to the prior_smoothing but it is uniform
		try:
			ncc = fcc.sum(axis=1).reshape(-1, 1)
		except ValueError:
			ncc = fcc.sum()
		fcc += (alpha / fcc.shape[0])

		return (fcc / ncc), (np.log(fcc) - np.log(ncc))

	def jelinek_mercer_smoothing(self, fcc, lambada=0.5, p_w=None):
		p_w_given_c = (fcc / fcc.sum(axis=1).reshape(-1, 1)) * (1 - lambada)
		p_w = ((fcc.sum(axis=0) / fcc.sum()).reshape(1, -1) * lambada) if p_w is None else p_w * lambada

		return (p_w + p_w_given_c), np.nan_to_num(np.log(p_w + p_w_given_c))

	def prior_smoothing(self, fcc, mu=0.95): # <-- I have no idea why this works, but it works better than anything else
		p_c = (fcc.sum(axis=1).reshape(-1, 1) / fcc.sum())

		enum = (fcc + (mu * p_c))
		denom = (fcc.sum(axis=1).reshape(-1, 1))
		p_w_given_c = enum / denom

		return p_w_given_c, np.log(enum) - np.log(denom)

	def dirichlet_smoothing(self, fcc, mu=0.95, p_w=None):
		#p_w = (fcc.sum(axis=1).reshape(-1, 1) / fcc.sum())
		p_w = (fcc.sum(axis=0) / fcc.sum()).reshape(1, -1) if p_w is None else p_w
		p_w_given_c = (fcc + (mu * p_w)) / ((fcc + mu).sum(axis=1).reshape(-1, 1))
		log_p_w_given_c = np.log(fcc + (mu * p_w)) - np.log((fcc + mu).sum(axis=1).reshape(-1, 1))

		return p_w_given_c, log_p_w_given_c

	def absolute_discounting(self, fcc, sigma=0.6, p_w=None):
		p_w = (fcc.sum(axis=0) / fcc.sum()).reshape(1, -1) if p_w is None else p_w

		num_unique_w = np.zeros((fcc.shape[0], 1))
		for row_idx in np.arange(fcc.shape[0]):
			num_unique_w[row_idx] = np.count_nonzero(fcc[row_idx])

		p_w_given_c = (np.maximum(fcc - sigma, 0.) + (np.dot((sigma * np.asmatrix(num_unique_w)), p_w))) / (fcc.sum(axis=1).reshape(-1, 1))

		return p_w_given_c, np.nan_to_num(np.log(p_w_given_c))

	def two_stage_smoothing(self, fcc, lambada=0.6, mu=100, p_w=None):
		p_w = (fcc.sum(axis=0) / fcc.sum()).reshape(1, -1) if p_w is None else p_w

		dirichlet_stage, _ = NaiveBayesSmoothing.dirichlet_smoothing(fcc, mu, p_w)

		p_w_given_c = ((1 - lambada) * dirichlet_stage) + (lambada * p_w)

		return p_w_given_c, np.nan_to_num(np.log(p_w_given_c))

	def calc_lidstone_damping_factor_tokens(self, vocab_size, n_tokens, alpha=1.):
		damping_magnitude_tokens = math.floor(math.log10(vocab_size * alpha)) / math.floor(math.log10(n_tokens))
		damping_factor_tokens = 10 ** damping_magnitude_tokens

		return damping_factor_tokens

	def calc_lidstone_damping_factor_vocab(self, vocab_size, sample_vocab_size, alpha=1.):
		damping_magnitude_vocab = math.floor(math.log10(vocab_size * alpha)) / math.floor(math.log10(sample_vocab_size))
		damping_factor_vocab = 10 ** damping_magnitude_vocab

		return damping_factor_vocab


class MultinomialCNB(MultinomialNB):

	def _update_feature_log_prob(self):
		"""Apply smoothing to raw counts and recompute log probabilities see Rennie (2003) for details on how
			to compute the complement counts"""
		super(MultinomialCNB, self)._update_feature_log_prob()

		self.complement_feature_log_prob_ = np.zeros(self.feature_log_prob_.shape)

		#mask_size = len(self.classes_) if (max(self.classes_) < len(self.classes_)) else max(self.classes_) + 1
		mask = np.ones(len(self.classes_), dtype=np.bool)

		for idx, c in enumerate(self.classes_):
			mask[idx] = False

			smoothed_complement_fc = (self.feature_count_[mask] + self.alpha).sum(axis=0)
			smoothed_complement_cc = smoothed_complement_fc.sum()

			self.complement_feature_log_prob_[idx] = (np.log(smoothed_complement_fc)
													- np.log(smoothed_complement_cc.reshape(-1, 1)))

			mask[:] = True

	def _joint_log_likelihood(self, X):
		"""Calculate the posterior log probability of the samples X"""
		X = check_array(X, accept_sparse='csr')
		return (self.class_log_prior_ -
				safe_sparse_dot(X, self.complement_feature_log_prob_.T))


class MultinomialWCNB(MultinomialCNB):
	def _update_feature_log_prob(self):
		"""Apply smoothing to raw counts and recompute log probabilities see Rennie (2003) for details on how
			to compute the complement counts"""
		super(MultinomialWCNB, self)._update_feature_log_prob()
		self.complement_feature_log_prob_ /= np.abs(self.complement_feature_log_prob_).sum(axis=1).reshape(-1, 1)


class MultinomialWNB(MultinomialNB):
	def _update_feature_log_prob(self):
		"""Apply smoothing to raw counts and recompute log probabilities see Rennie (2003) for details on how
			to compute the complement counts"""
		super(MultinomialWNB, self)._update_feature_log_prob()
		self.feature_log_prob_ /= np.abs(self.feature_log_prob_).sum(axis=1).reshape(-1, 1)


class SSLBernoulliNB(BernoulliNB):

	def __init__(self, alpha=1.0, binarize=.0, fit_prior=True,
				 class_prior=None):

		self.feature_prob_ = None
		self.alpha = alpha
		self.binarize = binarize
		self.fit_prior = fit_prior
		self.class_prior = class_prior

	def _update_feature_log_prob(self):
		super(SSLBernoulliNB, self)._update_feature_log_prob()

		self.feature_prob_ = np.exp(self.feature_log_prob_)

	def _fit_prior(self, y, sample_weight=None):

		labelbin = LabelBinarizer()
		Y = labelbin.fit_transform(y)
		self.classes_ = labelbin.classes_
		if Y.shape[1] == 1:
			Y = np.concatenate((1 - Y, Y), axis=1)

		# LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
		# We convert it to np.float64 to support sample_weight consistently;
		# this means we also don't have to cast X to floating point
		Y = Y.astype(np.float64)
		if sample_weight is not None:
			Y *= check_array(sample_weight).T

		class_prior = self.class_prior

		# Count raw events from data before updating the class log prior
		# and feature log probas
		n_effective_classes = Y.shape[1]
		self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
		self.class_count_ += Y.sum(axis=0)

		super(SSLBernoulliNB, self)._update_class_log_prior()

	def sfe_fit(self, X, y, Z): #TODO: Implement SFE/FM as Mixins?
		self.fit(X, y)

		smoothed_fc = self.feature_count_.sum(axis=0) + (self.alpha * len(self.classes_))
		smoothed_cc = (self.class_count_ + self.alpha * len(self.classes_)).sum()

		smoothed_fz = Z.sum(axis=0) + self.alpha

		pw_l = smoothed_fc / smoothed_cc
		pw_Z = smoothed_fz / smoothed_fz.sum()

		sfe_enum = np.multiply(((self.feature_prob_ * np.exp(self.class_log_prior_).reshape(len(self.class_count_), 1)) / pw_l), pw_Z)
		sfe_denom = sfe_enum.sum(axis=1)

		self.feature_log_prob_ = np.asarray(np.log(sfe_enum) - np.log(sfe_denom))

	def _fm_fit_binary(self, X, y, Z, clf, maxiter=100, tol=1e-10):

		clf.fit(X, y)

		smoothed_fc = clf.feature_count_.sum(axis=0) + (clf.alpha * len(clf.classes_))
		smoothed_cc = (clf.class_count_ + clf.alpha * len(clf.classes_)).sum()

		smoothed_fz = Z.sum(axis=0) + clf.alpha

		pt_L = smoothed_fc / smoothed_cc
		pw_Z = smoothed_fz / smoothed_fz.sum()

		# Shorthands K, l
		l = pt_L[0] / pt_L[1]
		K = pw_Z / pt_L[1]

		# Target Interval
		target_interval_max = pw_Z / pt_L[0]

		# Starting points
		x0 = [2, 3, 1.5, 4]

		# word count per class pre-computed
		n_w_pos_sum, n_w_neg_sum = clf.feature_count_[0].sum(), clf.feature_count_[1].sum()

		# Optimisation
		for i in xrange(clf.feature_count_.shape[1]):
			if (clf.feature_count_[0, i] > 0 and clf.feature_count_[1, i] > 0):
				n_w_pos = clf.feature_count_[0, i]
				n_w_neg = clf.feature_count_[1, i]
				n_not_w_pos = n_w_pos_sum - n_w_pos
				n_not_w_neg = n_w_neg_sum - n_w_neg

				j = 0
				opt_val = -1
				while (j < len(x0) and not (opt_val > 0 and opt_val <= target_interval_max[0, i])):
					try:
						opt_val = newton(self._optimise_feature_marginals, (target_interval_max[0, i] / x0[j]), args=(K[0, i], l, n_w_pos, n_w_neg, n_not_w_pos, n_not_w_neg), tol=tol, maxiter=maxiter)
					except RuntimeError: # failed to converge, returns NaN
						opt_val = -1
					finally:
						j += 1

				if (opt_val > 0 and opt_val <= target_interval_max[0, i]):
					clf.feature_prob_[0, i] = opt_val
					clf.feature_prob_[1, i] = (pw_Z[0, i] - (opt_val * pt_L[0])) / pt_L[1]

		# Re-normalise
		clf.feature_prob_ /= clf.feature_prob_.sum(axis=1).reshape(2, 1)
		clf.feature_log_prob_ = np.log(clf.feature_prob_)

		return clf

	def fm_fit(self, X, y, Z, maxiter=50, tol=1e-10, sample_weight=None):

		if (np.unique(y).shape[0] > 2):
			y_c = np.ones(X.shape[0], dtype=np.int8)

			estimators = []

			for c in range(np.unique(y).shape[0]):
				c_mask = y == c
				y_c[c_mask] = 0

				clf = clone(self)
				estimators.append(self._fm_fit_binary(X, y_c, Z, clf, maxiter=maxiter, tol=tol))

				y_c[:] = 1

			# Stuff together the estimators again
			self._fit_prior(y, sample_weight=sample_weight)

			n_classes = len(self.classes_)

			self.feature_count_ = np.zeros((n_classes, estimators[0].feature_count_.shape[1]))
			self.feature_prob_ = np.zeros((n_classes, estimators[0].feature_prob_.shape[1]))
			for c in range(len(self.classes_)):
				self.feature_count_[c, :] = estimators[c].feature_count_[0, :]
				self.feature_prob_[c, :] = estimators[c].feature_prob_[0, :]

			# Re-normalise
			self.feature_prob_ /= self.feature_prob_.sum(axis=1).reshape(len(self.classes_), 1)
			self.feature_log_prob_ = np.log(self.feature_prob_)
		else:
			self._fm_fit_binary(X, y, Z, maxiter=maxiter, tol=tol, clf=self)

	def _optimise_feature_marginals(self, p_w_pos, k, l, n_w_pos, n_w_neg, n_not_w_pos, n_not_w_neg):
		return 	(n_w_pos / p_w_pos) + \
				(n_not_w_pos / (p_w_pos - 1)) + \
				((n_w_neg * l) / ((l * p_w_pos) - k)) + \
				((l * n_not_w_neg) / ((l * p_w_pos) - k + 1))


class NaiveBayesClassifier(BaseEstimator, NaiveBayesSmoothingMixin):

	def __init__(self, fit_prior=True, class_prior=None, smoothing='lidstone_smoothing', smoothing_args=(1.,)):

		self.smoothing = smoothing
		self.smoothing_fn = smoothing if callable(smoothing) else wittgenstein.prepare_invocation_on_obj(self, smoothing)
		self.smoothing_args = smoothing_args
		self.classes_ = None
		self.priors_ = None
		self.log_priors_ = None
		self.feature_counts_ = None
		self.probs_ = None
		self.log_probs_ = None
		self.fit_prior = fit_prior
		self.class_prior = class_prior


	def fit(self, X, y):
		X = self._to_csr(X)

		self._fit_priors(y)

		self._fit_features(X, y)

	def predict(self, X):
		jll = self._joint_log_likelihood(X)

		return self.classes_[np.argmax(jll, axis=1)]

	def predict_proba(self, X):
		return np.exp(self.predict_log_proba(X))

	def predict_log_proba(self, X):
		jll = self._joint_log_likelihood(X)
		# normalize by P(x) = P(f_1, ..., f_n)
		log_prob_x = logsumexp(jll, axis=1)

		return jll - np.atleast_2d(log_prob_x).T

	def estimate_evidence(self, ranked_label_idx, Z, use_log_probs=True):

		E = self._estimate_log_prob_evidence(ranked_label_idx, Z) if use_log_probs else self._estimate_prob_evidence(ranked_label_idx, Z)

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

	def sfe_fit(self, X, y, Z): #TODO: Implement SFE/FM as Mixins?
		self.fit(X, y)

		pw_l, _ = self.smoothing_fn(self.feature_counts_.sum(axis=0), *self.smoothing_args)
		pw_Z, _ = self.smoothing_fn(Z, *self.smoothing_args)

		sfe_enum = np.multiply(((self.probs_ * self.priors_.reshape(len(self.classes_), 1)) / pw_l), pw_Z)
		sfe_denom = sfe_enum.sum(axis=1)

		self.probs_, self.log_probs_ = (sfe_enum / sfe_denom), np.log(sfe_enum) - np.log(sfe_denom)

	def _fm_fit_binary(self, X, y, Z, clf, maxiter=100, tol=1e-10):

		clf.fit(X, y)

		pt_L = clf.feature_counts_.sum(axis=1) / clf.feature_counts_.sum()
		pw_Z, _ = clf.smoothing_fn(Z, *clf.smoothing_args)

		# Shorthands K, l
		l = pt_L[0] / pt_L[1]
		K = pw_Z / pt_L[1]

		# Target Interval
		target_interval_max = pw_Z / pt_L[0]

		# Starting points
		x0 = [2, 3, 1.5, 4]

		# word count per class pre-computed
		n_w_pos_sum, n_w_neg_sum = clf.feature_counts_[0].sum(), clf.feature_counts_[1].sum()

		# Optimisation
		for i in xrange(clf.feature_counts_.shape[1]):
			if (clf.feature_counts_[0, i] > 0 and clf.feature_counts_[1, i] > 0):
				n_w_pos = clf.feature_counts_[0, i]
				n_w_neg = clf.feature_counts_[1, i]
				n_not_w_pos = n_w_pos_sum - n_w_pos
				n_not_w_neg = n_w_neg_sum - n_w_neg

				j = 0
				opt_val = -1
				while (j < len(x0) and not (opt_val > 0 and opt_val <= target_interval_max[0, i])):
					try:
						opt_val = newton(self._optimise_feature_marginals, (target_interval_max[0, i] / x0[j]), args=(K[0, i], l, n_w_pos, n_w_neg, n_not_w_pos, n_not_w_neg), tol=tol, maxiter=maxiter)
					except RuntimeError: # failed to converge, returns NaN
						opt_val = -1
					finally:
						j += 1

				if (opt_val > 0 and opt_val <= target_interval_max[0, i]):
					clf.probs_[0, i] = opt_val
					clf.probs_[1, i] = (pw_Z[0, i] - (opt_val * pt_L[0])) / pt_L[1]

		# Re-normalise
		clf.probs_ /= clf.probs_.sum(axis=1).reshape(2, 1)
		clf.log_probs_ = np.log(clf.probs_)

		return clf

	def fm_fit(self, X, y, Z, maxiter=50, tol=1e-10, fit_priors=True, class_priors=None):

		if (np.unique(y).shape[0] > 2):
			y_c = np.ones(X.shape[0], dtype=np.int8)

			estimators = []

			for c in range(np.unique(y).shape[0]):
				c_mask = y == c
				y_c[c_mask] = 0

				clf = clone(self)
				estimators.append(self._fm_fit_binary(X, y_c, Z, clf, maxiter=maxiter, tol=tol))

				y_c[:] = 1

			# Stuff together the estimators again
			self._fit_priors(y)

			n_classes = len(self.classes_)

			self.feature_counts_ = np.zeros((n_classes, estimators[0].feature_counts_.shape[1]))
			self.probs_ = np.zeros((n_classes, estimators[0].probs_.shape[1]))
			for c in range(len(self.classes_)):
				self.feature_counts_[c, :] = estimators[c].feature_counts_[0, :]
				self.probs_[c, :] = estimators[c].probs_[0, :]

			# Re-normalise
			self.probs_ /= self.probs_.sum(axis=1).reshape(len(self.classes_), 1)
			self.log_probs_ = np.log(self.probs_)
		else:
			self._fm_fit_binary(X, y, Z, maxiter=maxiter, tol=tol, clf=self)

	def _optimise_feature_marginals(self, p_w_pos, k, l, n_w_pos, n_w_neg, n_not_w_pos, n_not_w_neg):
		return 	(n_w_pos / p_w_pos) + \
				(n_not_w_pos / (p_w_pos - 1)) + \
				((n_w_neg * l) / ((l * p_w_pos) - k)) + \
				((l * n_not_w_neg) / ((l * p_w_pos) - k + 1))

	def _to_csr(self, X):
		if (sparse.isspmatrix_csr(X)):
			return X
		elif (sparse.issparse(X)):
			return X.tocsr()
		else: # i.e. numpy arrays/ndmatrices
			return sparse.csr_matrix(X)

	def _evidence_per_instance(self, ranked_label_idx, Z):
		E_log = np.zeros((ranked_label_idx.shape[0], 2))

		for row_idx, (rank_idx, z) in enumerate(itertools.izip(ranked_label_idx, Z)):
			E_log[row_idx, 0] = np.maximum(self.log_probs_[rank_idx[0], z.nonzero()[1]] - self.log_probs_[rank_idx[1], z.nonzero()[1]], 0.).sum()
			E_log[row_idx, 1] = np.maximum(self.log_probs_[rank_idx[1], z.nonzero()[1]] - self.log_probs_[rank_idx[0], z.nonzero()[1]], 0.).sum()

		return E_log

	# Margin of Confidence
	def _estimate_log_prob_evidence(self, ranked_label_idx, Z):
		# Evidence sets per instance for most-likely class and second most-likely class; (the un-indexed equations in section D. of Sharma & Bilgic (2013))
		''' Takes about a minute
		E_log = np.zeros((ranked_label_idx.shape[0], 2))

		import time
		print time.time()

		for row_idx, (rank_idx, z) in enumerate(itertools.izip(ranked_label_idx, Z)):
			E_log[row_idx, 0] = np.maximum(self.log_probs_[rank_idx[0], z.nonzero()[1]] - self.log_probs_[rank_idx[1], z.nonzero()[1]], 0.).sum()
			E_log[row_idx, 1] = np.maximum(self.log_probs_[rank_idx[1], z.nonzero()[1]] - self.log_probs_[rank_idx[0], z.nonzero()[1]], 0.).sum()

		print time.time()
		E = E_log.sum(axis=1)
		'''
		import time
		print time.time()
		MLC = np.zeros((ranked_label_idx.shape[0], Z.shape[1]))
		SMLC = np.zeros((ranked_label_idx.shape[0], Z.shape[1]))

		MLC[Z.nonzero()] = self.log_probs_[ranked_label_idx[:, 0]][Z.nonzero()]
		SMLC[Z.nonzero()] = self.log_probs_[ranked_label_idx[:, 1]][Z.nonzero()]

		E = np.maximum(MLC - SMLC, 0.).sum(axis=1)

		self.log_probs_[rank]
		print time.time()

		return E

	# Margin of Confidence
	def _estimate_prob_evidence(self, ranked_label_idx, Z):
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

	def _fit_priors(self, y):
		self.priors_ = np.bincount(y) / y.sum() if (self.fit_prior) else np.ones(np.unique(y).shape) * (1 / np.unique(y).shape[0])
		self.priors_ = self.priors_ if (self.class_prior is None) else self.class_prior
		self.classes_ = np.arange(self.priors_.shape[0])
		self.log_priors_ = np.nan_to_num(np.log(self.priors_))

	def _fit_features(self, X, y):
		self.feature_counts_ = np.zeros((self.priors_.shape[0], X.shape[1]))

		# Count Features
		for y_i in self.classes_:
			self.feature_counts_[y_i, :] = X[np.where(y == y_i)].sum(axis=0)

		#self.probs_, self.log_probs_ = self.smoothing_fn_(self.feature_counts_, *self.smoothing_args_)
		args = (self.feature_counts_,) + self.smoothing_args
		self.probs_, self.log_probs_ = self.smoothing_fn(*args)

	def _joint_log_likelihood(self, X):
		return safe_sparse_dot(X, self.log_probs_.T) + self.log_priors_