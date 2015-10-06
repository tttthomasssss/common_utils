__author__ = 'thomas'
import codecs
import os

from nltk.tokenize import word_tokenize
import numpy as np

from common import paths


def tokenize_kafka():
	with codecs.open(os.path.join(paths.get_dataset_path(), 'kafka', 'kafka_one_line_lc.txt'), 'rB', 'utf-8') as f:
		tokens = []
		for line in f:
			tokens = word_tokenize(line)
			print 'LEN:', len(tokens)
			vocab = sorted(set(tokens))
		print np.bincount(tokens)
	print 'Kafka Vocab Size=%d' % (len(vocab),)

	with codecs.open(os.path.join(paths.get_dataset_path(), 'kafka', 'kafka_vocab'), 'wB', 'utf-8') as f:
		for v in vocab:
			f.write(v + '\n')


if (__name__ == '__main__'):
	tokenize_kafka()