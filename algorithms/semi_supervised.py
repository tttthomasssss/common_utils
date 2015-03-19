__author__ = 'thomas'

from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB
import numpy as np


class EMWeightingSchemeMixin(object):

	def fit_alpha_weighting(self, Z, y_prob, clf, weights, use_max=False):
		return self._standard_max_fit(Z, y_prob, clf, weights) if use_max else self._standard_all_fit(Z, y_prob, clf, weights)

	def fit_adaptive_weighting(self, Z, y_prob, clf, weights, use_max=False):
		return self._adaptive_weighting_max_fit(Z, y_prob, clf, weights) if use_max else self._standard_all_fit(Z, y_prob, clf, np.repeat(weights.reshape(-1, 1), y_prob.shape[1], axis=1))

	def fit_prediction_weighting(self, Z, y_prob, clf, weights, use_max=False):
		return self._standard_max_fit(Z, y_prob, clf, weights) if use_max else self._standard_all_fit(Z, y_prob, clf, weights)

	def _adaptive_weighting_max_fit(self, Z, y_prob, clf, weights):
		clf.partial_fit(Z, np.argmax(y_prob, axis=1), sample_weight=weights)

		return clf

	def _standard_max_fit(self, Z, y_prob, clf, weights):
		y_Z = np.argmax(y_prob, axis=1)
		clf.partial_fit(Z, y_Z, sample_weight=weights[y_Z])

		return clf

	def _standard_all_fit(self, Z, y_prob, clf, weights):
		#y_Z = np.repeat(np.array(range(y_prob.shape[1])).reshape(1, -1), y_prob.shape[0], axis=0).T
		#clf.partial_fit(Z, y_Z, sample_weight=weights)
		for c in xrange(y_prob.shape[1]):
			y_c = np.full((y_prob.shape[0], 1), c)
			clf.partial_fit(Z, y_c, sample_weight=weights[:, c])

		return clf

	def alpha_weighting(self, y_prob, alpha=0.1):
		return np.full(y_prob.shape, alpha)

	def adaptive_weighting(self, y_prob, alpha=1., confidence_correction=0.):
		"""
		Calculates the weights based on how confident the classifier is in predicting the given instance
		Method is based on Conditional Entropy, which measures the uncertainty of the model
		1 - Conditional Entropy then represents the certainty
		:param y_prob: predicted probabilistic labels of current model
		:param alpha: additional multiplicative weighting factor
		:param confidence_correction: additional additive regularisation term (i.e. NB tends to over-estimate its confidence, hence this can be used to regularise its confidence)
		:return: weights for probabilistically labelled instances
		"""

		# Conditional Entropy of predicutions under current model
		H = -((y_prob * np.log(y_prob)).sum(axis=1))

		# Convert uncertainty to certainty
		H = np.nan_to_num(1 - H)

		# Re-normalise the weights
		H /= np.amax(H)

		# Apply additive regularisation
		H = np.maximum(H - confidence_correction, 0.)

		# Apply multiplicative weighting factor
		H *= alpha

		return H if np.max(H) > 0 else self.alpha_weighting(H)

	def prediction_weighting(self, y_prob, alpha=1., confidence_correction=0.):
		return np.maximum(y_prob - confidence_correction, 0.) * alpha


class ExpectationMaximization(BaseEstimator, EMWeightingSchemeMixin):
	def __init__(self, *args, **kwargs):
		classifier_args = kwargs.pop('classifier_args', [])
		classifier_kwargs = kwargs.pop('classifier_kwargs', {})
		self.clf_ = kwargs.pop('classifier', MultinomialNB(*classifier_args, **classifier_kwargs))

		super(ExpectationMaximization, self).__init__(*args, **kwargs)

	def fit(self, X, y, Z, max_iter=1, use_max=False, sample_weighting_scheme='alpha', **weighting_scheme_kwargs):
		"""
		Use Expectation-Maximization to train a Naive Bayes classifier
		:param X: labelled data
		:param y: labels
		:param Z: unlabelled data
		:param max_iter: maximum number of iterations in EM
		:param sample_weighting_scheme: weighting scheme for probabilistically labelled data, available options are 'alpha', 'adaptive' and 'prediction'
		"""
		weight_fn, fit_fn = self._select_fns(sample_weighting_scheme)

		# Step 1: Model Initialisation
		self.clf_.fit(X, y)

		for _ in xrange(max_iter):
			# Step 2: Generating probabilistic labels for the unlabelled data
			y_prob = self.clf_.predict_proba(Z)

			# Step 2.1: Apply weighting scheme
			weights = weight_fn(y_prob, **weighting_scheme_kwargs)

			# Step 3: Retrain classifier on all data
			#	Step 3.1 Retrain on labelled data
			self.clf_.partial_fit(X, y)

			# 	Step 3.2 Retrain on probabilistically labelled data
			self.clf_ = fit_fn(Z, y_prob, self.clf_, weights, use_max=use_max)


	def predict(self, X):
		return self.clf_.predict(X)

	def predict_proba(self, X):
		return self.clf_.predict_proba(X)

	def predict_log_proba(self, X):
		return self.clf_.predict_log_proba(X)

	def _select_fns(self, weighting_scheme):
		if (weighting_scheme == 'adaptive'):
			return self.adaptive_weighting, self.fit_adaptive_weighting
		elif (weighting_scheme == 'prediction'):
			return self.prediction_weighting, self.fit_prediction_weighting
		else:
			return self.alpha_weighting, self.fit_alpha_weighting