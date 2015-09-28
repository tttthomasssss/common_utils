__author__ = 'thomas'
import os

from gensim.test.test_doc2vec import read_su_sentiment_rotten_tomatoes

from common import paths

if (__name__ == '__main__'):
	phrases = read_su_sentiment_rotten_tomatoes(dirname=os.path.join(paths.get_dataset_path(), 'stanford_sentiment_treebank'))

	for phrase in phrases:
		if (phrase.split is not None and phrase.sentence_id is not None):
			print 'Split=[{}]; SentenceID=[{}]; Sentiment=[{}]: {}'.format(phrase.split, phrase.sentence_id, phrase.sentiment, phrase.words)
		#else:
		#	print '\t{}'.format(phrase.words)

	print 'Done'