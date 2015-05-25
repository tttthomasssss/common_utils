__author__ = 'thk22'
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from algorithms.semi_supervised import ExpectationMaximization
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

if (__name__ == '__main__'):
	em_test()