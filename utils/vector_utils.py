__author__ = 'thk22'
import collections
import csv
import gzip
import tarfile


def open_file(filename, mode='r', encoding='utf-8'):
	if (filename.endswith('tar.gz')):
		return tarfile.open(filename, mode)
	elif (filename.endswith('.gz')):
		return gzip.open(filename, mode, encoding=encoding)
	else:
		return open(filename, mode, encoding=encoding)


def apply_offset_path(path, offset):

	if (path.startswith(':')): # EPSILON
		offset_path = offset + path
	else: # TODO: What to do with invalid paths?
		head, *tail = path.split('\xbb')

		if ('_{}'.format(head) == offset or '_{}'.format(offset) == head):
			offset_path = '\xbb'.join(tail)
		else:
			offset_path = '\xbb'.join([offset, path])

	return offset_path

def create_offset_vector(vector, offset_path):
	# Translate from my notation to Dave's notation
	if (offset_path.startswith('!')):
		offset_path = '_' + offset_path[1:]

	offset_vector = {}

	for feat in vector.keys():
		offset_vector[apply_offset_path(feat, offset_path)] = vector[feat]

	return offset_vector

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

#---
#load in pre-filtered and (optionally) normalised vectors
#----
def load_csv_vectors(infile='', words=None, out_prefix='', mod_logging_freq=10000):
	vecs={}
	print ('Loading vectors from: {}'.format(infile))
	print ('Words of interest: {}'.format(words))
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
					vecs[entry] = add_vectors(vecs[entry], vector)
				else:
					vecs[entry] = vector

				if (words is not None):
					words.remove(entry)

			if (words is not None and len(words) <= 0): # Early stopping
				break

	print('{}Loaded {} vectors'.format(out_prefix, len(vecs.keys())))
	return vecs

#---
#add two vectors
#not used I think
#---
def add_vectors(self, avector, bvector):
	rvector=dict(avector)
	for feat in bvector.keys():
		rvector[feat] = rvector.get(feat,0)+bvector[feat]
	return rvector