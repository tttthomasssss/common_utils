__author__ = 'thk22'
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np

from algorithms.semi_supervised import ExpectationMaximization
from classifiers.naive_bayes import NaiveBayesSmoothing
from common import dataset_utils
from common import paths


def em_test():
	#X_train, y_train, X_test, y_test = dataset_utils.fetch_20newsgroups_dataset_vectorized(os.path.join(paths.get_dataset_path(), '20newsgroups'),
	#																					   use_tfidf=False, tf_normalisation=False, extraction_style='all')

	X_train, y_train, X_test, y_test, Z = dataset_utils.fetch_movie_reviews_dataset_vectorized(os.path.join(paths.get_dataset_path(), 'movie_reviews/aclImdb'),
																								use_tfidf=False, tf_normalisation=False, extraction_style='all')

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=True)
	em_pred = em.predict(X_test)

	print 'USE_MAX=True; ALPHA=0.1; MAX_ITER=1'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=False)
	em_pred = em.predict(X_test)

	print 'USE_MAX=False; ALPHA=0.1; MAX_ITER=1'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=True, alpha=1.)
	em_pred = em.predict(X_test)

	print 'USE_MAX=True; ALPHA=1.0; MAX_ITER=1'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=False, alpha=1.)
	em_pred = em.predict(X_test)

	print 'USE_MAX=False; ALPHA=1.0; MAX_ITER=1'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'
	####
	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=True, max_iter=5)
	em_pred = em.predict(X_test)

	print 'USE_MAX=True; ALPHA=0.1; MAX_ITER=5'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=False, max_iter=5)
	em_pred = em.predict(X_test)

	print 'USE_MAX=False; ALPHA=0.1; MAX_ITER=5'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=True, alpha=1., max_iter=5)
	em_pred = em.predict(X_test)

	print 'USE_MAX=True; ALPHA=1.0; MAX_ITER=5'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=False, alpha=1., max_iter=5)
	em_pred = em.predict(X_test)

	print 'USE_MAX=False; ALPHA=1.0; MAX_ITER=5'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'
	####
	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=True, max_iter=10)
	em_pred = em.predict(X_test)

	print 'USE_MAX=True; ALPHA=0.1; MAX_ITER=10'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=False, max_iter=10)
	em_pred = em.predict(X_test)

	print 'USE_MAX=False; ALPHA=0.1; MAX_ITER=10'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=True, alpha=1., max_iter=10)
	em_pred = em.predict(X_test)

	print 'USE_MAX=True; ALPHA=1.0; MAX_ITER=10'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'

	em = ExpectationMaximization()
	em.fit(X_train, y_train, Z, use_max=False, alpha=1., max_iter=10)
	em_pred = em.predict(X_test)

	print 'USE_MAX=False; ALPHA=1.0; MAX_ITER=10'
	print 'Accuracy:', accuracy_score(y_test, em_pred)
	print 'F1-Score:', f1_score(y_test, em_pred)
	print '------------------------------'


def em_multi_stage_test():
	#for kitkat in ['duggan-relevant-verdict-comment']:
	for kitkat in ['wow-misogyny-terms-uk-sw','amessagefromisustous-subversion','wow-abusive-or-not','clacton-master-filtered','floodingbtr','cleggfaragesearchrelevencytest','alessia-cameron2','floodingbtr5','isisenglishsubvertednot','bigdebatespeakersplitter-posneg','cameronboocheer','wow-rape-news','nato2subjects','bigdebatespeakersplitter','clacton-users-parties','boo-cheer','shellfiltration','duggan-relevant-verdict-comment','duggan-relevent-no-news','isis-kafir-english-relev','miliband','immigtation-sw-cameron','Immigration_extendedtermsrelevancy','duggan-main']:
		data = dataset_utils.fetch_ws_paper_dataset_vectorized(os.path.join(paths.get_dataset_path(), 'ws_paper'), kitkat, use_tfidf=False, tf_normalisation=False, extraction_style=None, ngram_range=(1, 2), load_vectorizer=True)
		X_train, y_train, X_test, y_test, Z, label_map, labelled_features, vec = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[8]

		alpha = 1 / NaiveBayesSmoothing.calc_lidstone_damping_factor_tokens(X_train.shape[1], X_train.sum())

		weighting = 'weighted' if np.unique(y_train).shape[0] > 2 else 'binary'

		print 'KITKAT:', kitkat

		mnb = MultinomialNB()
		mnb.fit(X_train, y_train)
		pred = mnb.predict(X_test)
		print 'MNB ACCURACY:', accuracy_score(y_test, pred)
		print 'MNB F1-SCORE:', f1_score(y_test, pred, pos_label=0, average=weighting)

		em_settles = ExpectationMaximization(classifier=MultinomialNB(alpha=alpha))
		em_settles.fit_settles(X_train, y_train, Z, labelled_features, label_map, vec.vocabulary_, use_max=True)
		pred = em_settles.predict(X_test)
		print 'EM ACCURACY:', accuracy_score(y_test, pred)
		print 'EM F1-SCORE:', f1_score(y_test, pred, pos_label=0, average=weighting)
		print '--------------------------------------------------------'

		'''
		print 'Unigrams...'
		vec = data[-2]
		labelled_features = data[6]

		for f_dict in labelled_features.itervalues():
			for f in f_dict:
				print f, f.strip() in vec.vocabulary_.keys()
		print '--------------------------------------------------------'

		print 'Bigrams...'

		data = dataset_utils.fetch_ws_paper_dataset_vectorized(os.path.join(paths.get_dataset_path(), 'ws_paper'), kitkat, use_tfidf=False, tf_normalisation=False, extraction_style='all', ngram_range=(1, 2), force_recreate_dataset=True)
		vec = data[-2]
		labelled_features = data[6]

		for f_dict in labelled_features.itervalues():
			for f in f_dict:
				print f, f.strip().replace('_', ' ') in vec.vocabulary_.keys()
		print '==========================================================\n'
		'''

		'''
		mnb = MultinomialNB()
		mnb.fit(X_train, y_train)
		pred = mnb.predict(X_test)
		print 'MNB ACCURACY:', accuracy_score(y_test, pred)
		print 'MNB F1-SCORE:', f1_score(y_test, pred, pos_label=0, average=weighting)

		# em_mc = ExpectationMaximization(classifier=MultinomialNB(alpha=alpha))
		# em_mc.fit_most_certain(X_train, y_train, Z, alpha=0.2) # 0.3 helped for a few
		# pred = em_mc.predict(X_test)
		# print 'EM MC ACCURACY:', accuracy_score(y_test, pred)
		# print 'EM MC F1-SCORE:', f1_score(y_test, pred, pos_label=0, average=weighting)

		em = ExpectationMaximization(classifier=MultinomialNB(alpha=alpha))
		em.fit_most_certain(X_train, y_train, Z)
		pred = em.predict(X_test)
		print 'EM ACCURACY:', accuracy_score(y_test, pred)
		print 'EM F1-SCORE:', f1_score(y_test, pred, pos_label=0, average=weighting)

		# em_ms = ExpectationMaximization(classifier=MultinomialNB(alpha=alpha))
		# em_ms.fit_multi_stage(X_train, y_train, Z, alpha=0.2) # 0.2 good for duggan-main
		# pred = em_ms.predict(X_test)
		# print 'EM MS ACCURACY:', accuracy_score(y_test, pred)
		# print 'EM MS F1-SCORE:', f1_score(y_test, pred, pos_label=0, average=weighting)

		em_mk = ExpectationMaximization(classifier=MultinomialNB(alpha=alpha))
		em_mk.fit_max_knowledge(X_train, y_train, Z, threshold=0.75)
		pred = em_mk.predict(X_test)
		print 'EM MK ACCURACY:', accuracy_score(y_test, pred)
		print 'EM MK F1-SCORE:', f1_score(y_test, pred, pos_label=0, average=weighting)

		# em_pwf = ExpectationMaximization(classifier=MultinomialNB(alpha=alpha))
		# em_pwf.fit_most_certain(X_train, y_train, Z, alpha=(X_train.shape[0] / Z.shape[0]))
		# pred = em_pwf.predict(X_test)
		# print 'EM PWF ACCURACY:', accuracy_score(y_test, pred)
		# print 'EM PWF F1-SCORE:', f1_score(y_test, pred, pos_label=0)
		#
		# em_mc_pwf = ExpectationMaximization(classifier=MultinomialNB(alpha=alpha))
		# em_mc_pwf.fit_most_certain(X_train, y_train, Z, alpha=(X_train.shape[0] / Z.shape[0]))
		# pred = em_mc_pwf.predict(X_test)
		# print 'EM MC PWF ACCURACY:', accuracy_score(y_test, pred)
		# print 'EM MC PWF F1-SCORE:', f1_score(y_test, pred, pos_label=0)

		print '--------------------------------------------------------------------------'
		'''

if (__name__ == '__main__'):
	#em_test()
	em_multi_stage_test()