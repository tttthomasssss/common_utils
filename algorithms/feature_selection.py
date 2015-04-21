from __future__ import division
__author__ = 'thk22'

import numpy as np
from scipy.stats import norm
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import binarize
from sklearn.utils import shuffle


def bns(X, y):
	X_pos, X_neg = X[np.where(y==0)], X[np.where(y==1)]
	P, N = X_pos.sum(axis=0) / X_pos.sum(), X_neg.sum(axis=0) / X_neg.sum()
	P_squashed, N_squashed = norm.ppf(norm.cdf(P)), norm.ppf(norm.cdf(N))

	F = np.squeeze(np.asarray(np.absolute(P_squashed - N_squashed)))

	return F, np.ones(F.shape)


def odds_ratio(X, y):
	X_pos, X_neg = X[np.where(y==0)], X[np.where(y==1)]
	P, N = X_pos.sum(axis=0) / X_pos.sum(), X_neg.sum(axis=0) / X_neg.sum()

	enum_log = np.log(P) + np.log(1 - N)
	denom_log = np.log((1 - P)) + np.log(N)

	F = np.squeeze(np.asarray(np.nan_to_num(enum_log) - np.nan_to_num(denom_log)))

	return F, np.ones(F.shape)


def information_gain(X, y):
	P_feat_present = X.sum(axis=0) / X.sum()
	P_feat_absent = 1.0 - P_feat_present

	IG = np.zeros((1, X.shape[1]), dtype=np.float64)

	for c in np.unique(y):
		idx_c = np.where(y==c)[0]
		X_c = X[idx_c]

		p_c = idx_c.shape[0] / y.shape[0]
		P_feat_present_given_c = X_c.sum(axis=0) / X_c.sum()
		P_feat_absent_given_c = 1 - P_feat_present_given_c

		# Case feature present (I_k = 1)
		IG_present = np.multiply((P_feat_present_given_c * p_c), np.log(np.divide(P_feat_present_given_c, P_feat_present)))

		# Case feature absent (I_k = 0)
		IG_absent = np.multiply((P_feat_absent_given_c * p_c), np.log(np.divide(P_feat_absent_given_c, P_feat_absent)))

		IG += (np.nan_to_num(IG_present) + np.nan_to_num(IG_absent))

	IG = np.squeeze(np.asarray(IG))

	return IG, np.ones(IG.shape)


def mutual_information(X, y):
	'''
	Mutual Information between a feature t of X and a class label c of y
	Calculated as defined by Li et al. (2009): A Framework of Feature Selection Methods for Text Categorization, 3.2 Mutual Information (MI)
		MI = log (A * n) / ((A + C) * (A + B))

		where
			A = the number of documents containing term t and belonging to class c
			B = the number of documents containing term t and NOT belonging to class c
			C = the number of documents NOT containing term t and belonging to class c
			n = the number of documents in the training set
	:param X:
	:param y:
	:return:
	'''

	X_bin = binarize(X)
	neg_mask = np.ones(X.shape[0], dtype=bool)

	labels = np.unique(y)

	A = np.zeros(labels.shape[0], X.shape[1])
	B = np.zeros(labels.shape[0], X.shape[1])
	C = np.zeros(labels.shape[0], X.shape[1])
	n = X.shape[0]

	for y_c in labels:
		y_c_mask = y == y_c
		neg_mask[y_c_mask] = False

		A[y_c] = X_bin[y_c_mask].sum(axis=0)
		B[y_c] = X_bin[neg_mask].sum(axis=0)
		C = X_bin[y_c_mask].shape[0] - A

		neg_mask[:] = True

	F = (np.log(A) + np.log(n)) - (np.log(A + C) + np.log(A + B))

	return F, np.ones(F.shape)


def chi_square(X, y):
	return chi2(X, y)


def document_frequency(X, y):
	F = np.zeros((1, X.shape[1]))
	for c in np.unique(y):
		X_c = X[np.where(y==c)]

		F += (X_c != 0).sum(axis=0)

	F = np.squeeze(np.asarray(F))

	return F, np.ones(F.shape)


def f_classif(X, y):
	return f_classif(X, y)


def random(X, y):
	F = np.arange(X.shape[1])

	return shuffle(F), np.ones(F.shape)

