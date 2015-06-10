from __future__ import division
import inspect

from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import binarize
import numpy as np

from . import instance_selection


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
		clf.partial_fit(Z, y_Z, sample_weight=weights[np.arange(y_Z.shape[0]), y_Z])

		return clf

	def _standard_all_fit(self, Z, y_prob, clf, weights):
		#y_Z = np.repeat(np.array(range(y_prob.shape[1])).reshape(1, -1), y_prob.shape[0], axis=0).T
		#clf.partial_fit(Z, y_Z, sample_weight=weights)
		for c in range(y_prob.shape[1]):
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
		classifier_args = kwargs.pop('classifier_args', ())
		classifier_kwargs = kwargs.pop('classifier_kwargs', {})
		#self.clf_ = kwargs.pop('classifier', MultinomialNB(*classifier_args, **classifier_kwargs))
		KhlavKalash = kwargs.pop('classifier', MultinomialNB)

		# If a class is passed, instantiate it, if an instance is passed deal with that!
		if (inspect.isclass(KhlavKalash)):
			self.clf_ = KhlavKalash(*classifier_args, **classifier_kwargs)
		else:
			self.clf_ = KhlavKalash

		super(ExpectationMaximization, self).__init__(*args, **kwargs)

	def fit_settles(self, X, y, Z, labelled_features, label_map, vocabulary, pseudo_count=50, co_occurrence_threshold=0.75, max_iter=1, use_max=False, sample_weighting_scheme='alpha', **weighting_scheme_kwargs):
		weight_fn, fit_fn = self._select_fns(sample_weighting_scheme)

		X_init = np.ones(X.shape)

		for label, feature_map in labelled_features.iteritems():
			for feature in feature_map.keys():
				cleaned_feat = feature.strip().replace('_', ' ')
				if (cleaned_feat in vocabulary):

					X_init[y == label_map[label], vocabulary.keys().index(cleaned_feat)] = pseudo_count
					feat_count = X[y == label_map[label], vocabulary.keys().index(cleaned_feat)].sum()

					for y_c in np.unique(y):
						y_c_feat_count = X[y == y_c, vocabulary.keys().index(cleaned_feat)].sum()
						if (y_c != label_map[label] and feat_count > 0 and y_c_feat_count >= (feat_count * co_occurrence_threshold)):
							X_init[y == y_c, vocabulary.keys().index(cleaned_feat)] = pseudo_count

		# Step 1: Model Initialisation
		#self.clf_.fit(X + X_init, y)
		#self.clf_ = MultinomialNB(fit_prior=False)
		self.clf_.fit(X_init, y)

		for _ in range(max_iter):
			# Step 2: Generating probabilistic labels for the unlabelled data
			y_prob = self.clf_.predict_proba(Z)

			# Step 2.1: Apply weighting scheme
			weights = weight_fn(y_prob, **weighting_scheme_kwargs)

			# Step 3: Retrain classifier on all data
			#	Step 3.1 Retrain on labelled data
			self.clf_.partial_fit(X + X_init, y)
			#self.clf_ = MultinomialNB()
			#self.clf_.fit(X + X_init, y)

			# 	Step 3.2 Retrain on probabilistically labelled data
			self.clf_ = fit_fn(Z, y_prob, self.clf_, weights, use_max=use_max)

	def fit_max_knowledge(self, X, y, Z, threshold=0.5, max_iter=1, use_max=False, sample_weighting_scheme='alpha', **weighting_scheme_kwargs):
		weight_fn, fit_fn = self._select_fns(sample_weighting_scheme)

		# Step 1: Model Initialisation
		self.clf_.fit(X, y)

		for _ in range(max_iter):
			# Step 2: Generating probabilistic labels for the unlabelled data
			y_prob = self.clf_.predict_proba(Z)

			K = binarize(Z).sum(axis=1) / X.shape[1]

			# Normalise K
			K /= np.amax(K)

			Z_idx = np.squeeze(np.asarray((np.where(K > threshold)[0])))

			# Step 2.1: Apply weighting scheme
			weights = weight_fn(y_prob, **weighting_scheme_kwargs)

			# Step 3: Retrain classifier on all data
			#	Step 3.1 Retrain on labelled data
			self.clf_.partial_fit(X, y)

			# 	Step 3.2 Retrain on probabilistically labelled data
			self.clf_ = fit_fn(Z[Z_idx], y_prob[Z_idx], self.clf_, weights[Z_idx], use_max=use_max)

	def fit_most_certain(self, X, y, Z, threshold=0.5, max_iter=1, use_max=False, sample_weighting_scheme='alpha', **weighting_scheme_kwargs):
		weight_fn, fit_fn = self._select_fns(sample_weighting_scheme)

		# Step 1: Model Initialisation
		self.clf_.fit(X, y)

		for _ in range(max_iter):
			# Step 2: Generating probabilistic labels for the unlabelled data
			y_prob = self.clf_.predict_proba(Z)

			# Conditional Entropy of predicutions under current model
			H = -((y_prob * np.log(y_prob)).sum(axis=1))

			# Convert uncertainty to certainty
			H = np.nan_to_num(1 - H)

			# Re-normalise the weights
			H /= np.amax(H)

			XXX = np.sort(H)

			# print np.where(XXX <= 0.5)[0].shape
			# print np.where(XXX > 0.5)[0].shape
			# print np.where(XXX > 0.75)[0].shape
			# print np.where(XXX > 0.9)[0].shape
			# print np.where(XXX > 0.95)[0].shape
			# print np.where(XXX > 0.99)[0].shape
			# print np.where(XXX > 0.999)[0].shape
			# print np.where(XXX > 0.9999999)[0].shape
			# print np.where(XXX > 0.9999999999)[0].shape
			# print np.where(XXX == 1.)[0].shape

			# import os
			# from matplotlib import pyplot as plt
			# plt.figure(figsize=(18, 9), dpi=600, facecolor='w', edgecolor='k')
			# plt.xlabel('X')
			# plt.ylabel('Certainty')
			# plt.title('Certainty over instances')
			# plot_name = 'certainty_over_instances.png'
			# plt.grid(True)
			# plt.hold(True)
			# plt.ylim(0., 1.)
			# plt.margins(0.01)
			# x = np.arange(XXX.shape[0])
			# plt.plot(XXX)
			# plt.savefig(os.path.join('/Volumes/LocalDataHD/thk22/DevSandbox/InfiniteSandbox/_results/em_multi_stage', plot_name))
			# plt.close()

			# TODO: Second loop, only select highly MSU ones first, then gradually adding in LSU ones

			# Step 2.1: Apply weighting scheme
			weights = weight_fn(y_prob, **weighting_scheme_kwargs)

			# Step 3: Retrain classifier on all data
			#	Step 3.1 Retrain on labelled data
			self.clf_.partial_fit(X, y)

			#idx = np.where(H < threshold)
			idx = np.where(H == 1.)

			# 	Step 3.2 Retrain on probabilistically labelled data
			self.clf_ = fit_fn(Z[idx], y_prob[idx], self.clf_, weights[idx], use_max=use_max)

	# TODO: Put in a Mixin(?)
	def fit_multi_stage(self, X, y, Z, stages=[1., 0.999], max_iter=1, use_max=False, sample_weighting_scheme='alpha', **weighting_scheme_kwargs):
		weight_fn, fit_fn = self._select_fns(sample_weighting_scheme)

		# Step 1: Model Initialisation
		self.clf_.fit(X, y)

		for _ in range(max_iter):
			# Step 2: Generating probabilistic labels for the unlabelled data
			y_prob = self.clf_.predict_proba(Z)

			# Conditional Entropy of predicutions under current model
			H = -((y_prob * np.log(y_prob)).sum(axis=1))

			# Convert uncertainty to certainty
			H = np.nan_to_num(1 - H)

			# Re-normalise the weights
			H /= np.amax(H)

			# Step 2.1: Apply weighting scheme
			weights = weight_fn(y_prob, **weighting_scheme_kwargs)

			# Step 3: Retrain classifier on all data
			#	Step 3.1 Retrain on labelled data
			self.clf_.partial_fit(X, y)

			idx = np.where(H >= stages[0])
			idx_other = np.where(H < stages[0])

			C = Z[idx]
			y_c = y_prob[idx]
			w_c = weights[idx]

			# Step 3.2 Retrain on probabilistically labelled data (most certain only)
			self.clf_ = fit_fn(Z[idx], y_prob[idx], self.clf_, weights[idx], use_max=use_max)

			# Re-classify rest
			y_prob = self.clf_.predict_proba(Z[idx_other])

			# Conditional Entropy of predicutions under current model
			H = -((y_prob * np.log(y_prob)).sum(axis=1))

			# Convert uncertainty to certainty
			H = np.nan_to_num(1 - H)

			# Re-normalise the weights
			H /= np.amax(H)

			# Step 2.1: Apply weighting scheme
			weights = weight_fn(y_prob, **weighting_scheme_kwargs)

			# Step 3: Retrain classifier on all data
			#	Step 3.1 Retrain on labelled data
			self.clf_.partial_fit(X, y)
			self.clf_ = fit_fn(C, y_c, self.clf_, w_c, use_max=use_max)

			idx = np.where(H >= stages[0])
			idx_other = np.where(H < stages[0])

			# Step 3.2 Retrain on probabilistically labelled data (most certain only)
			self.clf_ = fit_fn(Z[idx], y_prob[idx], self.clf_, weights[idx], use_max=use_max)



	def fit(self, X, y, Z, max_iter=1, use_max=False, sample_weighting_scheme='alpha', init='fit', **weighting_scheme_kwargs):
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
		if (init is not None):
			self.clf_.fit(X, y)

		for _ in range(max_iter):
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