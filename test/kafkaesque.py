from __future__ import division
__author__ = 'thomas'
import codecs
import csv
import os

from nltk.tokenize import word_tokenize
from scipy import sparse
from scipy.spatial import distance
from sklearn.datasets import fetch_20newsgroups
import joblib
import numpy as np

from common import paths
from wort.vsm import VSMVectorizer


def vectorize_kafka():

	docs = [
		'i sat on a table',
		'the cat sat on the mat.',
		'the pizza sat next to the table',
		'a green curry sat under the chair'
	]

	vec = VSMVectorizer(window_size=2, min_frequency=2)
	M_ppmi = vec.fit_transform(docs)

	with open(os.path.join(paths.get_dataset_path(), 'kafka', 'kafka_one_line_lc.txt'), mode='r', encoding='utf-8') as f:
		#vec = VSMVectorizer(window_size=5, cds=0.75, svd=300, svd_eig_weighting=0.5, sppmi_shift=5)
		vec = VSMVectorizer(window_size=5)
		M_ppmi = vec.fit_transform(f)

		print ('PPMI Matrix created!')

	words = filter(lambda w: True if w in vec.inverted_index_.keys() else False, ['manager', 'director', 'clerk', 'innocent', 'judge', 'court', 'lawyer', 'law', 'josef', 'gregor', 'animal', 'samsa', 'trial', 'sister', 'father', 'mother', 'office', 'coat', 'work', 'fear', 'love', 'hate', 'manner', 'money', 'suit', 'custom', 'house', 'visitor'])

	for w in words:
		idx = vec.inverted_index_[w]

		min_dist = np.inf
		min_idx = -1

		for i in range(M_ppmi.shape[0]):
			if (i != idx):
				curr_dist = distance.cosine(M_ppmi[idx].A, M_ppmi[i].A)

				if (curr_dist < min_dist):
					min_idx = i
					min_dist = curr_dist

		print('\t[SIM=%.4f] WORD=%s; MOST SIMILAR=%s' % (min_dist, w, vec.index_[min_idx]))


def tokenize_kafka():

	# Load Tokens
	with codecs.open(os.path.join(paths.get_dataset_path(), 'kafka', 'kafka_one_line_lc.txt'), 'rB', 'utf-8') as f:
		tokens = []
		for line in f:
			tokens = word_tokenize(line)
			vocab = sorted(set(tokens))
	print ('Kafka Vocab Size=%d' % (len(vocab),))

	# Write Vocab
	with codecs.open(os.path.join(paths.get_dataset_path(), 'kafka', 'kafka_vocab'), 'wB', 'utf-8') as f:
		for v in vocab:
			f.write(v + '\n')

	index = dict(enumerate(vocab))
	inverted_index = dict(zip(index.values(), index.keys()))

	tokens_idx = np.array(list(map(lambda w: inverted_index[w], tokens)))

	bins = np.bincount(tokens_idx)
	p_w = bins / bins.sum()
	'''
	print np.where(bins > 50)[0].shape
	print np.where(bins > 25)[0].shape
	print np.where(bins > 20)[0].shape
	print np.where(bins > 10)[0].shape
	print np.where(bins > 5)[0].shape
	print np.where(bins > 3)[0].shape
	print np.where(bins > 2)[0].shape
	print np.where(bins > 1)[0].shape
	'''

	# Co-occurrence matrix with varying window sizes
	window_sizes = [1, 3, 5, 10]

	for s in window_sizes:
		#fname = 'kafka_cooccurrence_count_size-%s.dat' % (s,)
		#memarr = np.memmap(os.path.join(paths.get_dataset_path(), 'kafka', fname), dtype=np.uint32, mode='w+', shape=(len(vocab), len(vocab)))
		#memarr[:] = np.zeros((len(vocab), len(vocab)))

		print('Constructing Co-occurrence Matrix with window_size=%d' % (s,))

		fname = 'kafka_cooccurrence_count_size-%s_sparse' % (s,)
		#M = np.zeros((len(vocab), len(vocab)), dtype=np.uint16)
		M = sparse.dok_matrix((len(vocab), len(vocab)), dtype=np.uint32)

		for idx in index.keys():
			occurrences = np.where(tokens_idx==idx)[0]
			for occ_idx in occurrences:
				cooccurrence_window = np.concatenate((tokens_idx[occ_idx - s:occ_idx], tokens_idx[occ_idx + 1:occ_idx + 1 + s]))

				#memarr[idx, cooccurrence_window] += 1
				M[idx, cooccurrence_window] += 1

		#memarr.flush()
		#del memarr
		joblib.dump(M, os.path.join(paths.get_dataset_path(), 'kafka', fname))

		# PLMI, PPMI & PNPMI calculated according to https://www.aclweb.org/anthology/W/W14/W14-1502.pdf
		print('\tConstructing PPMI, PLMI, PNPMI & SPPMI Matrices...')

		fname_ppmi = 'kafka_cooccurrence_ppmi_size-%s_sparse' % (s,)
		fname_plmi = 'kafka_cooccurrence_plmi_size-%s_sparse' % (s,)
		fname_pnpmi = 'kafka_cooccurrence_pnpmi_size-%s_sparse' % (s,)
		fname_sppmi = 'kafka_cooccurrence_sppmi_size-%s_sparse' % (s,)

		'''
		M_ppmi = np.zeros(M.shape, dtype=np.float64)
		M_plmi = np.zeros(M.shape, dtype=np.float64)
		M_pnpmi = np.zeros(M.shape, dtype=np.float64)
		M_sppmi = np.zeros(M.shape, dtype=np.float64)
		'''
		M_ppmi = sparse.lil_matrix(M.shape, dtype=np.float64)
		M_plmi = sparse.lil_matrix(M.shape, dtype=np.float64)
		M_pnpmi = sparse.lil_matrix(M.shape, dtype=np.float64)
		M_sppmi = sparse.lil_matrix(M.shape, dtype=np.float64)

		# Joint Probability for all co-occurrences, P(w, c) = P(c | w) * P(w) = P(w | c) * P(c)
		P_w_c = (M / M.sum(axis=1)).A * p_w.reshape(-1, 1)

		# Perform PPMI weighting
		for idx in index.keys():
			#row = np.where(M[idx, :] > 0)[0]
			row = (M[idx, :] > 0).indices

			# Marginals for context
			p_c = p_w[row]

			# PMI
			pmi = np.log(P_w_c[idx, row]) - (np.log(p_w[idx]) + np.log(p_c))

			# PLMI
			plmi = P_w_c[idx, row] * pmi

			# PNPMI
			pnpmi = (P_w_c[idx, row] * (1 / -(np.log(p_c)))) * pmi

			# SPPMI
			sppmi = pmi - np.log(s) # out of convenience re-use the window size as the shifting factor...

			M_ppmi[idx, row] = np.maximum(0, pmi)
			M_plmi[idx, row] = np.maximum(0, plmi)
			M_pnpmi[idx, row] = np.maximum(0, pnpmi)
			M_sppmi[idx, row] = np.maximum(0, sppmi)

		joblib.dump(sparse.csr_matrix(M_ppmi), os.path.join(paths.get_dataset_path(), 'kafka', fname_ppmi))
		joblib.dump(sparse.csr_matrix(M_plmi), os.path.join(paths.get_dataset_path(), 'kafka', fname_plmi))
		joblib.dump(sparse.csr_matrix(M_pnpmi), os.path.join(paths.get_dataset_path(), 'kafka', fname_pnpmi))
		joblib.dump(sparse.csr_matrix(M_sppmi), os.path.join(paths.get_dataset_path(), 'kafka', fname_sppmi))

	# Dump Helper Files
	joblib.dump(index, os.path.join(paths.get_dataset_path(), 'kafka', 'index'))
	joblib.dump(inverted_index, os.path.join(paths.get_dataset_path(), 'kafka', 'inverted_index'))
	joblib.dump(vocab, os.path.join(paths.get_dataset_path(), 'kafka', 'vocab'))

	# TODO: SVD based on http://www.aclweb.org/anthology/Q/Q15/Q15-1016.pdf, esp. chapter 7, practical recommendations
	# Context Window Weighting
	# Subsampling
	# Context Distribution Smoothing
	# Hellinger PCA


def kafka_most_similar():

	weighting_schemes = ['ppmi', 'plmi', 'pnpmi', 'sppmi']
	fname_template = 'kafka_cooccurrence_%s_size-5'

	index = joblib.load(os.path.join(paths.get_dataset_path(), 'kafka', 'index'))
	inverted_index = joblib.load(os.path.join(paths.get_dataset_path(), 'kafka', 'inverted_index'))
	vocab = joblib.load(os.path.join(paths.get_dataset_path(), 'kafka', 'vocab'))

	words = filter(lambda w: True if w in vocab else False, ['manager', 'director', 'clerk', 'innocent', 'judge', 'court', 'lawyer', 'law', 'josef', 'gregor', 'animal', 'samsa', 'trial', 'sister', 'father', 'mother', 'office', 'coat', 'work', 'fear', 'love', 'hate', 'manner', 'money', 'suit', 'custom', 'house', 'visitor'])

	for weighting_scheme in weighting_schemes:
		fname = fname_template % (weighting_scheme,)
		M = joblib.load(os.path.join(paths.get_dataset_path(), 'kafka', fname))
		print('WEIGHTING SCHEME: %s' % (weighting_scheme,))
		for w in words:
			idx = inverted_index[w]

			min_dist = np.inf
			min_idx = -1

			for i in xrange(M.shape[1]):
				if (i != idx):
					curr_dist = distance.cosine(M[idx], M[i])

					if (curr_dist < min_dist):
						min_idx = i
						min_dist = curr_dist

			print('\t[SIM=%.4f] WORD=%s; MOST SIMILAR=%s' % (min_dist, w, index[min_idx]))

def wikipedia_iterator():
	from corpus_readers.wikipedia import WikipediaReader

	binary = [True, False]
	weightings = ['ppmi', 'sppmi', 'pnpmi', 'plmi']
	window_sizes = [3, 5, 10]
	min_frequencies = [20, 50, 100]

	with open(os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews.csv')) as csv_file:
		#csv_reader = csv.reader(csv_file)
		#vec = VSMVectorizer(window_size=5, use_memmap=True, memmap_path=os.path.join(paths.get_dataset_path(), 'wikipedia'))
		vec = VSMVectorizer(window_size=5)

		M_ppmi = vec.fit_transform(csv_file)

		print('SIZE: %r' % (M_ppmi.shape,))

		#joblib.dump(M_ppmi, os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_ppmi_size-5'))

	'''
	r = WikipediaReader(os.path.join(paths.get_dataset_path(), 'wikipedia', 'wikipedia_utf8_filtered_20pageviews.csv'))
	token_count = 0

	for idx, tokens in enumerate(r):
		if (idx % 1000 == 0):
			print('%d lines processed; current token count = %d...' % (idx, token_count))

		token_count += len(tokens)

	print ('NUM_LINES=%d; NUM_TOKENS=%d' % (idx, token_count))
	'''

if (__name__ == '__main__'):
	#tokenize_kafka()
	#kafka_most_similar()
	#wikipedia_iterator()
	vectorize_kafka()
