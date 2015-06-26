__author__ = 'thomas'

from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import MetaEstimatorMixin
from sklearn.utils import check_random_state


class Bumper(ClassifierMixin, MetaEstimatorMixin):
	def __init__(self, estimator, n_bumps=50, random_state=None):
		self.estimator_ = estimator
		self.n_bumps_ = n_bumps
		self.random_state_ = random_state
		self.best_estimator_ = None

	def fit(self, X, y, sample_weight=None, verbose=False, sample_size=1.):
		random_state = check_random_state(self.random_state_)
		n_samples, n_features = X.shape

		self.best_estimator_ = None
		best_score = None
		best_estimator = None

		for i in range(self.n_bumps_):
			self._log(verbose, 'Iteration %d of %d' % (i + 1, self.n_bumps_))

			idx = random_state.randint(0, n_samples - 1, int(round(n_samples * sample_size)))

			estimator = clone(self.estimator_)
			estimator.fit(X[idx], y[idx])

			score = estimator.score(X, y)
			if (score > best_score):
				self._log(verbose, '\tFound better estimator; score=%f' % (score,))
				best_score = score
				best_estimator = estimator

		self.best_estimator_ = best_estimator
		return self

	def predict(self, X):
		return self.best_estimator_.predict(X)

	def _log(self, verbose, text):
		if (verbose):
			print(text)