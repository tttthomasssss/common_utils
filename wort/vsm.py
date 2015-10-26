__author__ = 'thomas'
import array
import os

from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from sparsesvd import sparsesvd
import numpy as np

# TODO: SVD based on http://www.aclweb.org/anthology/Q/Q15/Q15-1016.pdf, esp. chapter 7, practical recommendations
	# Context Window Weighting
	# Subsampling
	# Normalisation
	# Hellinger PCA
	# NMF as an alternative to SVD
	# Support min_df and max_df
class VSMVectorizer(BaseEstimator, VectorizerMixin):
	def __init__(self, window_size, weighting='ppmi', min_frequency=0, lowercase=True, stop_words=None, encoding='utf-8',
				 max_features=None, preprocessor=None, tokenizer=None, analyzer='word', binary=False, sppmi_shift=1,
				 token_pattern=r'(?u)\b\w\w+\b', decode_error='strict', strip_accents=None, input='content',
				 ngram_range=(1, 1), cds=1, svd=None, svd_eig_weighting=1, add_context_vectors=True,
				 use_memmap=False, memmap_path=None):

		self.window_size = window_size
		self.weighting = weighting
		self.min_frequency = min_frequency
		self.lowercase = lowercase
		self.stop_words = stop_words
		self.encoding = encoding
		self.max_features = max_features
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.analyzer = analyzer
		self.binary = binary
		self.sppmi_shift = sppmi_shift
		self.token_pattern = token_pattern
		self.decode_error = decode_error
		self.strip_accents = strip_accents
		self.use_memmap = use_memmap
		self.memmap_path = memmap_path
		self.input = input
		self.ngram_range = ngram_range
		self.cds = cds
		self.svd_dims = svd
		self.svd_eig_weighting = svd_eig_weighting
		self.add_context_vectors = add_context_vectors

		self.inverted_index_ = {}
		self.index_ = {}
		self.p_w_ = None
		self.vocab_count_ = 0
		self.temp_tokens_store_ = None
		self.M_ = None
		self.T_ = None

	def _delete_from_vocab(self, W, idx):
		W = np.delete(W, idx)

		for i in idx:
			item = self.index_[i]
			del self.inverted_index_[item]
			#del self.index_[i]

		return W

	def _extract_vocabulary(self, raw_documents):
		analyse = self.build_analyzer()

		n_vocab = -1
		w = array.array('i')

		# Tempfile to hold _all_ tokens (to speed up the sliding window process later on)
		#tokens = [] too memory intensive

		# Extract Vocabulary
		# todo gensim corpus/Dictionary object or a sklearn.DictVectorizer(???)

		buffer = []


		for doc in raw_documents:
			for feature in analyse(doc):
				'''
				idx = self.inverted_index_.get(feature, n_vocab + 1)

				# Build vocab
				if (idx > n_vocab):
					n_vocab += 1
					self.inverted_index_[feature] = n_vocab
					w.append(1)
				else:
					w[idx] += 1
				'''

				if (feature in self.inverted_index_):
					idx = self.inverted_index_[feature]
					w[idx] += 1 # TODO: Do we need the bloody counts????
				else:
					n_vocab += 1
					self.inverted_index_[feature] = n_vocab
					w.append(1)


		# Vocab was used for indexing (hence, started at 0 for the first item (NOT init!)), so has to be incremented by 1
		# to reflect the true vocab count
		n_vocab += 1
		# Create Index
		print('MANUAL VOCAB COUNT: {}'.format(n_vocab))
		print('VOCAB COUNT: {}'.format(len(self.inverted_index_.keys())))
		#self.index_ = dict(enumerate(self.inverted_index_.keys()))
		#n_vocab = len(self.index_.keys())

		W = np.array(w, dtype=np.uint32)

		# todo gensim.Dictionary.filter_extremes (???)
		# Filter for Frequency
		if (not self.binary and self.min_frequency > 0):
			idx = np.where(W < self.min_frequency)[0]
			W = self._delete_from_vocab(W, idx)

			n_vocab -= len(idx)

		# Max Features Filter
		if (self.max_features is not None and self.max_features < n_vocab):
			idx = np.argpartition(-W)[self.max_features + 1:]
			W = self._delete_from_vocab(W, idx)

			n_vocab -= len(idx)

		self.p_w_ = W / W.sum()
		self.vocab_count_ = n_vocab

	def _construct_cooccurrence_matrix_mmap(self):
		self.M_ = np.memmap(os.path.join(self.memmap_path, 'M.dat'), dtype=np.uint32, mode='w+', shape=(self.vocab_count_, self.vocab_count_))
		self.M_[:] = np.zeros((self.vocab_count_, self.vocab_count_))

		for idx in self.index_.keys():
			occurrences = np.where(self.temp_tokens_store_==idx)[0]
			for occ_idx in occurrences:
				cooccurrence_window = np.concatenate((self.temp_tokens_store_[occ_idx - self.window_size:occ_idx], self.temp_tokens_store_[occ_idx + 1:occ_idx + 1 + self.window_size]))

				self.M_[idx, cooccurrence_window] += 1

		self.M_.flush()

	def _construct_cooccurrence_matrix(self):
		self.M_ = sparse.dok_matrix((self.vocab_count_, self.vocab_count_), dtype=np.uint32)



		''' To mem heavy
		for idx in self.index_.keys():
			occurrences = np.where(self.temp_tokens_store_==idx)[0]
			for occ_idx in occurrences:
				cooccurrence_window = np.concatenate((self.temp_tokens_store_[occ_idx - self.window_size:occ_idx], self.temp_tokens_store_[occ_idx + 1:occ_idx + 1 + self.window_size]))

				self.M_[idx, cooccurrence_window] += 1
		'''
	def _binarise(self):
		if (self.binary): # todo move outside function
			self.M_ = np.minimum(self.M, 1)

	def _apply_weight_option(self, pmi, P_w_c, p_c, idx, row):
		if (self.weighting == 'ppmi'):
			return pmi
		elif (self.weighting == 'plmi'):
			return P_w_c[idx, row] * pmi
		elif (self.weighting == 'pnpmi'):
			return (P_w_c[idx, row] * (1 / -(np.log(p_c)))) * pmi
		elif (self.weighting == 'sppmi'):
			return pmi - np.log(self.sppmi_shift)

	def _transform_memmap(self):
		self.T_ = np.memmap(os.path.join(self.memmap_path, '%s.dat' % (self.weighting.upper(),)), dtype=np.uint32, mode='w+', shape=(self.vocab_count_, self.vocab_count_))
		self.T_[:] = np.zeros((self.vocab_count_, self.vocab_count_), dtype=np.float64)

		# Joint Probability for all co-occurrences, P(w, c) = P(c | w) * P(w) = P(w | c) * P(c)
		P_w_c = self.M_ / self.M_.sum(axis=1) * self.p_w_.reshape(-1, 1)

		# Perform weighting
		for idx in self.index_.keys():
			row = np.where(self.M_[idx, :] > 0)[0]

			# Marginals for context
			p_c = self.p_w_[row] ** self.cds

			# PMI
			pmi = np.log(P_w_c[idx, row]) - (np.log(self.p_w_[idx]) + np.log(p_c))

			# Apply PMI variant (e.g. PPMI, SPPMI, PLMI or PNPMI)
			tpmi = self._apply_weight_option(pmi, self.weighting, p_c, idx, row)

			self.T_[idx, row] = np.maximum(0, tpmi)

		return self.T_

	def _transform(self):
		self.T_ = sparse.lil_matrix(self.M_.shape, dtype=np.float64)

		# Joint Probability for all co-occurrences, P(w, c) = P(c | w) * P(w) = P(w | c) * P(c)
		P_w_c = (self.M_ / self.M_.sum(axis=1)).A * self.p_w_.reshape(-1, 1)

		# Perform weighting
		for idx in self.index_.keys():
			row = (self.M_[idx, :] > 0).indices

			# Marginals for context (with optional context distribution smoothing)
			p_c = self.p_w_[row] ** self.cds

			# PMI
			pmi = np.log(P_w_c[idx, row]) - (np.log(self.p_w_[idx]) + np.log(p_c))

			# Apply PMI variant (e.g. PPMI, SPPMI, PLMI or PNPMI)
			# todo discoutils has an OK-ish PPMI implementation
			tpmi = self._apply_weight_option(pmi, self.weighting, p_c, idx, row)

			self.T_[idx, row] = np.maximum(0, tpmi)

		# CSR!!!
		self.T_ = self.T_.tocsr()

		# Apply SVD
		if self.svd_dims:
			Ut, S, Vt = sparsesvd(self.T_.tocsc(), self.svd_dims)

			# Perform Context Weighting
			S = sparse.csr_matrix(np.diag(S ** self.svd_eig_weighting))

			W = sparse.csr_matrix(Ut.T).dot(S)
			V = sparse.csr_matrix(Vt.T).dot(S)

			# Add context vectors
			if (self.add_context_vectors):
				self.T_ = W + V
			else:
				self.T_ = W

		return self.T_

	def fit(self, raw_documents, y=None):
		self._extract_vocabulary(raw_documents)

		# Construct Co-Occurrence Matrix
		if (self.use_memmap):
			self._construct_cooccurrence_matrix_mmap()
		else:
			self._construct_cooccurrence_matrix()

		self.temp_tokens_store_ = None

		# Apply Binarisation
		self._binarise() # todo make optional

		return self

	def transform(self, raw_documents):
		# todo move to a different class or rename?
		# Apply the weighting transformation
		if (self.use_memmap):
			return sparse.csr_matrix(self._transform_memmap())
		else:
			return self._transform()

	def fit_transform(self, raw_documents, y=None):
		self.fit(raw_documents)
		return self.transform(raw_documents)

