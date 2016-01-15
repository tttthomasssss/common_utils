__author__ = 'thk22'
import collections
import csv
import gzip
import os
import string
import tarfile

from common import paths
from scipy import sparse
import dill
import numpy as np
import joblib


def open_file(filename, mode='r', encoding='utf-8'):
	if (filename.endswith('tar.gz')):
		return tarfile.open(filename, mode)
	elif (filename.endswith('.gz')):
		return gzip.open(filename, mode, encoding=encoding)
	else:
		return open(filename, mode, encoding=encoding)


def apply_offset_path(path, offset, incompatible_paths='ignore'):
	'''
	:param path: dependency path feature
	:param offset: offset path
	:param incompatible_paths: pass 'strict' to exclude incompatible paths (double negative path) or 'ignore' to leave them in
	:return: offset_path or None if the path is incompatible and `incompatible_paths='strict'`
	'''

	#print('PATH: {}'.format(path))
	#print('\tOFFSET: {}'.format(offset))

	if (path.startswith(':')): # EPSILON
		offset_path = offset + path
	elif (path.startswith('_') and incompatible_paths == 'strict'):
		offset_path = None
	else: # TODO: What to do with invalid paths?
		head, feat = path.rsplit(':', 1)
		parts = head.split('\xbb')

		if ('_{}'.format(parts[0]) == offset or '_{}'.format(offset) == parts[0]):
			offset_path = '{}:{}'.format('\xbb'.join(parts[1:]), feat)
		else:
			offset_path = '\xbb'.join([offset, path])

	return offset_path


def create_offset_vector(vector, offset_path, incompatible_paths='ignore'):
	# Translate from my notation to Dave's notation
	if (offset_path.startswith('!')):
		offset_path = '_' + offset_path[1:]

	v = {}

	for feat in vector.keys():
		new_feat_path = apply_offset_path(feat, offset_path, incompatible_paths=incompatible_paths)

		if (new_feat_path is not None):
			v[new_feat_path] = vector[feat]

	return v


def collapse_offset_vector(offset_vector, offset_path=None):
	reduced_offset_vector = collections.defaultdict(float) # TODO: for experiment, need reduction/collapsing step after offset!!!!
	for key in offset_vector.keys():
		head, feat = key.rsplit(':', 1)

		parts = head.split('\xbb')
		if (len(parts) > 1):
			offset = parts[0] if offset_path is None else offset_path
			if (offset[1:] == parts[1] or offset == parts[1][1:]):
				new_key = '{}:{}'.format('\xbb'.join(parts[2:]), feat)
				print('\tReducing Key={} to={}'.format(key, new_key))
			else:
				new_key = key
		else:
			new_key = key

		reduced_offset_vector[new_key] += offset_vector[key]

		return reduced_offset_vector


def load_vector_cache(vector_in_file):
	if (vector_in_file.endswith('.dill')):
		return dill.load(open(vector_in_file, 'rb'))
	elif (vector_in_file.endswith('.joblib')):
		return joblib.load(vector_in_file)
	else:
		raise NotImplementedError


def oov_check(in_file, words, out_prefix, mod_logging_freq=10000):
	print ('Loading vectors from: {}'.format(in_file))
	print ('Words of interest: {}'.format(words))
	with open_file(in_file, 'rt', encoding='utf-8') as in_stream:
		for line_idx, line in enumerate(in_stream):
			if (line_idx % mod_logging_freq == 0): print('{}Reading line {}; Words left to find: {}'.format(out_prefix, line_idx, len(words) if words is not None else -1))

			line = line.rstrip().split('\t') # Line ends with a tab

			entry = line[0]
			if (entry in words):

				words.remove(entry)

			if (words is not None and len(words) <= 0): # Early stopping
				break

	return words


def find_vector_indices(in_file, words, out_prefix, mod_logging_freq=10000):
	vector_index = {}
	print ('Loading vectors from: {}'.format(in_file))
	print ('Words of interest: {}'.format(words))
	with open_file(in_file, 'rt', encoding='utf-8') as in_stream:
		for line_idx, line in enumerate(in_stream):
			if (line_idx % mod_logging_freq == 0): print('{}Reading line {}; Words left to find: {}'.format(out_prefix, line_idx, len(words) if words is not None else -1))

			line = line.rstrip().split('\t') # Line ends with a tab

			entry = line[0]
			if (entry in words):
				vector_index[entry] = line_idx

				words.remove(entry)

			if (words is not None and len(words) <= 0): # Early stopping
				break

	return vector_index


def vectorise_csv_vectors(in_file, out_path, key_path, logging, words=None, out_prefix='', mod_logging_freq=3000):
	logging.info('Loading vectors from: {}'.format(in_file))
	logging.info('Words of interest: {}'.format(words))
	with open_file(in_file, 'rt', encoding='utf-8') as in_vectors:
		keys = joblib.load(key_path)
		keys_list = list(keys)
		n_keys = len(keys)
		lines = 0
		oov = set()

		for line in in_vectors:
			lines += 1
			if (lines % mod_logging_freq == 0): logging.info('{}Reading line {}; Words left to find: {}'.format(out_prefix, lines, len(words) if words is not None else -1))

			line = line.rstrip().split('\t') # Line ends with a tab

			entry = line[0]

			# Fancy shit to avoid the dump falling over becasue `entry` is or contains punctuation
			trans_table = dict(map(lambda x: (x[1], '__PUNCT__' + str(x[0])), enumerate(string.punctuation, 666)))
			fname = os.path.join(out_path, '.'.join([entry.translate(str.maketrans(trans_table)), 'joblib']))

			if (not os.path.exists(fname) and (words is None or entry in words)):
				v = np.zeros((n_keys,), dtype=float)
				features = line[1:]

				index = 0
				while len(features) > 0:
					index += 1

					freq = float(features.pop())
					feat = features.pop()

					if (feat in keys):
						idx = keys_list.index(feat)
						v[idx] = freq
					else:
						oov.add(feat)
						logging.info('[WARNING] - {} not found in keyset!'.format(feat))

				# Store array in sparse format on disk
				joblib.dump(sparse.csr_matrix(v), fname)

				if (words is not None):
					words.remove(entry)

			if (words is not None and len(words) <= 0): # Early stopping
				break

	logging.info('{} were not found in the keyset ({} entries)!'.format(oov, len(oov)))


#---
#load in pre-filtered and (optionally) normalised vectors
#----
def load_csv_vectors(infile='', words=None, out_prefix='', mod_logging_freq=10000):
	vecs={}
	print('Words of interest: {}'.format(words))
	print('Thats {} words.'.format(len(words) if words is not None else 'all'))
	print('Loading vectors from: {}'.format(infile))
	with open_file(infile, 'rt', encoding='utf-8') as instream:
		#csv_reader = csv.reader(instream, delimiter='\t')
		lines = 0
		for line in instream: #csv_reader:
			lines += 1
			if (lines % mod_logging_freq == 0): print('{}Reading line {}; Words left to find: {}'.format(out_prefix, lines, len(words) if words is not None else -1))

			line = line.rstrip().split('\t') # Line ends with a tab

			entry = line[0]
			#print(entry)

			if (words is None or entry in words):
				vector = {}
				features = line[1:]

				index = 0
				while len(features) > 0:
					index += 1

					freq = features.pop()
					feat = features.pop()

					#print str(index)+'\t'+feat+'\t'+str(freq)
					try:
						freq = float(freq)
						vector[feat] = freq
					except ValueError:
						print('{}Error: {}\t{}\t{}\n'.format(out_prefix, index, feat, freq))
						features = features + list(feat)

				if entry in vecs.keys():
					print('{}Found {} in keys! Going to merge the two by adding!'.format(out_prefix, entry))
					vecs[entry] = add_vectors(vecs[entry], vector)
				else:
					vecs[entry] = vector

				if (words is not None):
					words.remove(entry)

			if (words is not None and len(words) <= 0): # Early stopping
				break

	print('{}Loaded {} vectors; {} vectors are oov.'.format(out_prefix, len(vecs.keys()), len(words)))
	return vecs


def collect_keys(in_file, out_path, logging):
	full_keys = set()
	paths_only = set()

	coarse_suffix = '_coarse' if 'coarse' in in_file else ''
	filtered_suffix = '_filtered' if 'filtered' in in_file else ''

	with open_file(in_file, 'rt', encoding='utf-8') as in_vectors:
		for idx, line in enumerate(in_vectors, 1):
			logging.info('Processing line {}...'.format(idx))
			line = line.rstrip().split('\t') # Line ends with a tab

			features = line[1:]

			while len(features) > 0:
				_ = features.pop()
				feat = features.pop()

				p, _ = split_path_from_word(feat)

				full_keys.add(feat)
				paths_only.add(p)

	logging.info('Dumping full_keys to {}; len={}...'.format( os.path.join(out_path, 'full_keys{}{}.joblib'.format(coarse_suffix, filtered_suffix)), len(full_keys)))
	joblib.dump(full_keys, os.path.join(out_path, 'full_keys{}{}.joblib'.format(coarse_suffix, filtered_suffix)))
	logging.info('Dumped full_keys!')

	logging.info('Dumping path_only to {}; len={}...'.format(os.path.join(out_path, 'paths_only{}{}.joblib'.format(coarse_suffix, filtered_suffix)), len(paths_only)))
	joblib.dump(paths_only, os.path.join(out_path, 'paths_only{}{}.joblib'.format(coarse_suffix, filtered_suffix)))
	logging.info('Dumped paths_only!')


def filter_vector(vector, min_count, min_features, logging, max_depth=np.inf, force_keep=False):
	filtered_vector = {}
	filtered_feat_count = 0
	feat_count = 0

	for feat, freq in vector.items():
		feat_count += 1

		if (freq >= min_count and len(feat.split('\xbb')) <= max_depth):
			filtered_feat_count += 1
			filtered_vector[feat] = freq

	logging.info('original feat count={}; new feat count={}'.format(feat_count, filtered_feat_count))

	if (force_keep): # Sometimes we want to _really_ keep the vector (e.g. in the case where its a target vector for comparison)
		len_vec = len(filtered_vector)
		if (len_vec < min_features and len_vec > 0):
			logging.warning('length of filtered vector[={}] < min_features[={}] > 0, keeping filtered vector as is.'.format(len_vec, min_features))
			return filtered_vector
		elif (len_vec < min_features and len_vec <= 0):
			logging.warning('length of filtered vector[={}] <= 0; returning original, unfiltered vector!'.format(len_vec))
			return vector
		elif (len_vec > min_features):
			return filtered_vector
		else:
			logging.error('Bad stuff is happening [len_vec={}; min_features={}]! Raising error!'.format(len_vec, min_features))
			raise RuntimeError
	else:
		return filtered_vector if (len(filtered_vector) >= min_features) else {}


def filter_csv_vectors(in_file, out_file, min_count, min_features, logging, keep_vectors=set(), normalise=False):
	with open_file(in_file, 'rt', encoding='utf-8') as in_vectors, open_file(out_file, 'wt', encoding='utf-8') as out_vectors:
		vec_count = 0
		filtered_vec_count = 0
		for idx, line in enumerate(in_vectors, 1):
			if idx % 3000 == 0: logging.info('Converting line {}...'.format(idx))
			vec_count += 1

			feat_count = 0
			filtered_feat_count = 0
			line = line.lower().rstrip().split('\t') # Line ends with a tab

			entry = line[0]

			features = line[1:]
			filtered_vector = {}
			temp_vector = {}
			keep_vectors_flag = False

			# Check if filtered vector is in keep_features and whether it conforms to the requirments
			if (entry in keep_vectors):
				logging.info('\t {} in keep_vectors...'.format(entry))
				keep_vectors_flag = True
				while len(features) > 0:
					feat_count += 1
					freq = float(features.pop())
					feat = features.pop()

					temp_vector[feat] = freq
					if (freq >= min_count):
						filtered_feat_count += 1
						filtered_vector[feat] = freq

				if (len(filtered_vector) < min_features):
					logging.info('\tFiltered vector is smaller than min_length (len={}...)'.format(len(filtered_vector)))
					if (len(filtered_vector) > 0):
						logging.info('\tFiltered vector length > 0, so will be represented as a filtered vector!')
					else:
						logging.info('\tFiltered vector length <= 0, so vector for word={} will be represented as an unfiltered vector!'.format(entry))
						filtered_vector = temp_vector
			else:
				while len(features) > 0:
					feat_count += 1
					freq = float(features.pop())
					feat = features.pop()

					if (freq >= min_count):
						filtered_feat_count += 1
						filtered_vector[feat] = freq

			logging.info('\tProcessing Filtered Vector of length={}...'.format(len(filtered_vector)))

			# Renormalise vectors
			if (normalise):
				logging.info('\tNormalising vector...')
				total = sum(filtered_vector.values())

				for feat, val in filtered_vector.items():
					filtered_vector[feat] /= total
				logging.info('\tFinished Normalising!')

			# Write Features
			if (len(filtered_vector) >= min_features or keep_vectors_flag):
				filtered_vec_count += 1
				logging.info('\tStarting to write vector for entry={} [old_feat_count={}; new_feat_count={}]...'.format(entry, feat_count, filtered_feat_count))
				out_vectors.write(entry + '\t')
				for k, v in filtered_vector.items():
					out_vectors.write(k + '\t' + str(v) + '\t')
				out_vectors.write('\n')
				logging.info('\tVector written to disk!')
			else:
				logging.info('\tSkipping vectors due to too few features: {} (min_features={})'.format(len(filtered_vector), min_features))

	logging.info('Conversion finished! Original number of vectors={}; filtered number={}!'.format(vec_count, filtered_vec_count))


def convert_csv_vectors(in_file, out_file, conversion_map, normalise=False):
	with open_file(in_file, 'rt', encoding='utf-8') as in_vectors, open_file(out_file, 'wt', encoding='utf-8') as out_vectors:
		for idx, line in enumerate(in_vectors, 1):
			if (idx % 3000 == 0): print('\tConverting line {}...'.format(idx))

			line = line.rstrip().split('\t') # Line ends with a tab

			entry = line[0]
			out_vectors.write(entry + '\t')

			features = line[1:]
			converted_features = collections.defaultdict(float)
			while len(features) > 0:
				freq = float(features.pop())
				feat = features.pop()

				p, w = split_path_from_word(feat)

				converted_deps = []
				for f in p.split('\xbb'):
					if (f.startswith('_')):
						converted_dep = conversion_map[f[1:]]
						converted_deps.append('_{}'.format(converted_dep))
					else:
						converted_deps.append(conversion_map[f])

				converted_feature = ':'.join(['\xbb'.join(converted_deps), w])

				# Collapse new duplicates (several distinct dep paths may map to the same grouped path)
				converted_features[converted_feature] += freq

			# Renormalise vectors
			if (normalise):
				total = sum(converted_features.values())

				for feat, val in converted_features.items():
					converted_features[feat] /= total

			# Write Features
			for k, v in converted_features.items():
				out_vectors.write(k + '\t' + str(v) + '\t')
			out_vectors.write('\n')
	print('\tConversion finished!')



def split_path_from_word(event, universal_deps=open(os.path.join(paths.get_dataset_path(), 'stanford_universal_deps.csv'), 'r').read().strip().split(',')):
	if (event.count(':') > 1):
		path = []
		word = []

		for dep in reversed(event.split('\xbb')):
			d = dep if not dep.startswith('_') else dep[1:]

			if (':' in d):
				subpath = []
				for sub_dep in reversed(dep.split(':')):
					sd = sub_dep if not sub_dep.startswith('_') else sub_dep[1:]
					if (sd in universal_deps):
						subpath.insert(0, sub_dep)
					else:
						word.insert(0, sub_dep)

				# Handle words that happen to have the same signifier as a dependency
				# The unfortunate case of a word containing a ':', where both parts of
				# the word happen to be signifiers of a dependency is currently unhandled
				# The hope is that if this case happens, it happens so infrequently as not
				# to matter at all
				if (len(word) <= 0):
					word.insert(0, subpath.pop())

				path.insert(0, ':'.join(subpath))
			else:
				path.insert(0, dep)
		return '\xbb'.join(path), ':'.join(word)
	else:
		return event.rsplit(':', 1)


#---
#add two vectors
#not used I think
#---
def add_vectors(avector, bvector):
	rvector = dict(avector)
	for feat in bvector.keys():
		rvector[feat] = rvector.get(feat, 0) + bvector[feat]
	return rvector