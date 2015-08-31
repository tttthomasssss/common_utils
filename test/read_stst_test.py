__author__ = 'thomas'
import os

from gensim.test.test_doc2vec import read_su_sentiment_rotten_tomatoes

from common import paths

if (__name__ == '__main__'):
	read_su_sentiment_rotten_tomatoes(dirname=os.path.join(paths.get_dataset_path(), 'stanford_sentiment_treebank'))