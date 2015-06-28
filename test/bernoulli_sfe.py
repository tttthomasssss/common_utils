from __future__ import division
__author__ = 'thomas'
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB

from classifiers.naive_bayes import NaiveBayesSmoothing
from classifiers.naive_bayes import SSLBernoulliNB
from common import dataset_utils
from common import paths

def bernoulli_sfe_test():
	data = dataset_utils.fetch_ws_paper_dataset_vectorized(os.path.join(paths.get_dataset_path(), 'ws_paper'), 'boo-cheer',
														   tf_normalisation=False, use_tfidf=False, extraction_style='all',
														   binarize=True)

	X_train = data[0]
	y_train = data[1]
	X_test = data[2]
	y_test = data[3]
	Z = data[4]

	nb_smoothing_alpha_binary = 1 / NaiveBayesSmoothing.calc_lidstone_damping_factor_tokens(X_train.shape[1], X_train.sum())

	bnb = BernoulliNB(alpha=nb_smoothing_alpha_binary)
	bnb.fit(X_train, y_train)
	pred = bnb.predict(X_test)
	print 'BNB ACC:', accuracy_score(y_test, pred)
	print 'BNB F1S:', f1_score(y_test, pred, average='micro')

	bnb = SSLBernoulliNB(alpha=nb_smoothing_alpha_binary)
	bnb.sfe_fit(X_train, y_train, Z)
	pred = bnb.predict(X_test)
	print 'SSLBNB ACC:', accuracy_score(y_test, pred)
	print 'SSLBNB F1S:', f1_score(y_test, pred, average='micro')

if (__name__ == '__main__'):
	bernoulli_sfe_test()