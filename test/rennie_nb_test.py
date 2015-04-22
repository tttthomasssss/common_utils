__author__ = 'thomas'
import os

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from classifiers.naive_bayes import MultinomialCNB
from classifiers.naive_bayes import MultinomialWCNB
from classifiers.naive_bayes import MultinomialWNB
from common import dataset_utils
from common import paths


def cnb_test():
	#X_train, y_train, X_test, y_test, Z = dataset_utils.fetch_movie_reviews_dataset_vectorized(os.path.join(paths.get_dataset_path(), 'movie_reviews', 'aclImdb'), use_tfidf=False, tf_normalisation=False)
	X_train, y_train, X_test, y_test = dataset_utils.fetch_20newsgroups_dataset_vectorized(os.path.join(paths.get_dataset_path(), '20newsgroups'), use_tfidf=False, tf_normalisation=False)

	cnb = MultinomialCNB()
	mnb = MultinomialNB()
	wnb = MultinomialWNB()
	wcnb = MultinomialWCNB()

	cnb.fit(X_train, y_train)
	pred = cnb.predict(X_test)

	print('CNB Accuracy: %f' % (accuracy_score(y_test, pred)))
	print('CNB F1-Score: %f' % (f1_score(y_test, pred)))

	mnb.fit(X_train, y_train)
	pred = mnb.predict(X_test)

	print('MNB Accuracy: %f' % (accuracy_score(y_test, pred)))
	print('MNB F1-Score: %f' % (f1_score(y_test, pred)))

	wcnb.fit(X_train, y_train)
	pred = wcnb.predict(X_test)

	print('WCNB Accuracy: %f' % (accuracy_score(y_test, pred)))
	print('WCNB F1-Score: %f' % (f1_score(y_test, pred)))

	wnb.fit(X_train, y_train)
	pred = wnb.predict(X_test)

	print('WNB Accuracy: %f' % (accuracy_score(y_test, pred)))
	print('WNB F1-Score: %f' % (f1_score(y_test, pred)))

if (__name__ == '__main__'):
	cnb_test()