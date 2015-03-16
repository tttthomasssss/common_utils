__author__ = 'thomas'

from scipy import sparse

import numpy as np


def safe_add(X, y, return_sparse_fmt=False):
	if (sparse.issparse(X) and np.isscalar(y)): # THIS IS SUPER INEFFICIENT!!!!
		return X + np.full(X.shape, y) if not return_sparse_fmt else sparse.csr_matrix(X + np.full(X.shape, y))
	elif (not sparse.issparse(X)):
		X += y
		return X if not return_sparse_fmt else sparse.csr_matrix(X)
	else:
		return X + y