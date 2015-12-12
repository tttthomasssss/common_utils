__author__ = 'thk22'
from time import gmtime
from time import strftime
import bz2
import collections
import csv
import json
import glob
import gzip
import pickle
import os
import string

#from bs4 import BeautifulSoup
#from discoutils import stanford_utils
from gensim.models import Word2Vec
from gensim.test.test_doc2vec import read_su_sentiment_rotten_tomatoes
from scipy.io import loadmat
from sklearn.datasets import fetch_20newsgroups
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import numpy as np

from corpus_hacks.aptemod import AptemodIndex
from corpus_hacks.rcv1 import RCV1Index

from . import paths
from utils import path_utils
from utils import vector_utils


# TODO: WSD Dataset resource: http://www.cs.cmu.edu/~mfaruqui/suite.html

__REUTERS_APTEMOD__ = 'aptemod'
__REUTERS_RCV1__ = 'rcv1'
__20_NEWSGROUPS__ = '20newsgroups'
__20_NEWSGROUPS_BINARISED__ = '20newsgroups_binarised'
__SMS_SPAM_COLLECTION__ = 'smsspamcollection'
__MOVIE_REVIEWS__ = 'movie_reviews'
__TWITTER_CAMERON__ = 'cameron'
__TWITTER_CAMERON_2__ = 'cameron-2'
__TWITTER_DUGGAN__ = 'duggan'
__TWITTER_DUGGAN_MAIN__ = 'duggan-main'
__TWITTER_DUGGAN_VERDICT__ = 'duggan-verdict'
__TWITTER_IMMIGRATION_BILL__ ='immigration-bill'
__TWITTER_IMMIGRATION_UK__ ='immigration-uk'
__TWITTER_IMMIGRATION_EXT__ ='immigration-ext'
__TWITTER_THEVOICEUK__ ='thevoiceuk'
__TOY_EXAMPLE__ = 'toy-example'
__WEBKB__ = 'webkb'
__SNOWDEN__ = 'snowden'
__SLUTHOE__ = 'sluthoe'

__DEFAULT_PATHS__ = {
	__REUTERS_APTEMOD__: os.path.join(paths.get_dataset_path(), 'aptemod/reuters21578'),
	#__REUTERS_RCV1__: os.path.join(local_paths.get_dataset_path(), 'RCV1.txt.bz2'),
	__REUTERS_RCV1__: os.path.join(paths.get_dataset_path(), 'rcv1'),
	__SMS_SPAM_COLLECTION__:os.path.join(paths.get_dataset_path(), 'smsspamcollection/SMSSpamCollection'),
	__MOVIE_REVIEWS__: os.path.join(paths.get_dataset_path(), 'movie_reviews/aclImdb'),
	__TWITTER_CAMERON__: os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__TWITTER_CAMERON_2__: os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__TWITTER_DUGGAN__: os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__TWITTER_DUGGAN_MAIN__: os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__TWITTER_DUGGAN_VERDICT__: os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__TWITTER_IMMIGRATION_BILL__:os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__TWITTER_IMMIGRATION_UK__:os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__TWITTER_IMMIGRATION_EXT__:os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__TWITTER_THEVOICEUK__:os.path.join(paths.get_dataset_path(), 'twitter_fyp'),
	__20_NEWSGROUPS__: os.path.join(paths.get_dataset_path(), '20newsgroups'),
	__20_NEWSGROUPS_BINARISED__: os.path.join(paths.get_dataset_path(), '20newsgroups'),
	__TOY_EXAMPLE__: os.path.join(paths.get_dataset_path(), 'toy_corpus'),
	__WEBKB__: os.path.join(paths.get_dataset_path(), 'webkb'),
    __SNOWDEN__: os.path.join(paths.get_dataset_path(), 'method51_classif'),
    __SLUTHOE__: os.path.join(paths.get_dataset_path(), 'method51_classif')
}

def _create_cache_path(file_path, vectorizer):
	tail, head = os.path.split(file_path)
	prefix = 'tfidf' if 'tfidf' in vectorizer.__str__().lower() else 'count'
	head = '%s-%s' % (prefix, head)
	cache_path = os.path.join(tail, head)

	return cache_path

def fetch_vectorized_datasets_for_keys(keys, **kwargs):
	data = collections.defaultdict(dict)

	if (__REUTERS_APTEMOD__ in keys):
		train_data, train_labels, test_data, test_labels = fetch_aptemod_dataset_vectorized(__DEFAULT_PATHS__[__REUTERS_APTEMOD__])
		data[__REUTERS_APTEMOD__]['train_data'] = train_data
		data[__REUTERS_APTEMOD__]['train_labels'] = train_labels
		data[__REUTERS_APTEMOD__]['test_data'] = test_data
		data[__REUTERS_APTEMOD__]['test_labels'] = test_labels

	if (__REUTERS_RCV1__ in keys):
		train_data, train_labels = fetch_rcv1_dataset_vectorized(__DEFAULT_PATHS__[__REUTERS_RCV1__])
		data[__REUTERS_RCV1__]['train_data'] = train_data
		data[__REUTERS_RCV1__]['train_labels'] = train_labels

	if (__20_NEWSGROUPS__ in keys):
		train_data, train_labels, test_data, test_labels = fetch_20newsgroups_dataset_vectorized(__DEFAULT_PATHS__[__20_NEWSGROUPS__])
		data[__20_NEWSGROUPS__]['train_data'] = train_data
		data[__20_NEWSGROUPS__]['train_labels'] = train_labels
		data[__20_NEWSGROUPS__]['test_data'] = test_data
		data[__20_NEWSGROUPS__]['test_labels'] = test_labels

	if (__20_NEWSGROUPS_BINARISED__ in keys):
		categories = kwargs.get('twenty_newsgroups_bin_category_keys', [])
		for cat in categories:
			train_data, train_labels, test_data, test_labels = fetch_20newsgroups_dataset_binarised_and_vectorized(__DEFAULT_PATHS__[__20_NEWSGROUPS_BINARISED__], cat)
			key = '%s_%s' % (__20_NEWSGROUPS_BINARISED__, cat)
			data[key]['train_data'] = train_data
			data[key]['train_labels'] = train_labels
			data[key]['test_data'] = test_data
			data[key]['test_labels'] = test_labels

	if (__SMS_SPAM_COLLECTION__ in keys):
		train_data, train_labels = fetch_sms_spam_collection_dataset_vectorized(__DEFAULT_PATHS__[__SMS_SPAM_COLLECTION__])
		data[__SMS_SPAM_COLLECTION__]['train_data'] = train_data
		data[__SMS_SPAM_COLLECTION__]['train_labels'] = train_labels

	if (__MOVIE_REVIEWS__ in keys):
		train_data, train_labels, test_data, test_labels, unlabelled_data = fetch_movie_reviews_dataset_vectorized(__DEFAULT_PATHS__[__MOVIE_REVIEWS__])
		data[__MOVIE_REVIEWS__]['train_data'] = train_data
		data[__MOVIE_REVIEWS__]['train_labels'] = train_labels
		data[__MOVIE_REVIEWS__]['test_data'] = test_data
		data[__MOVIE_REVIEWS__]['test_labels'] = test_labels
		data[__MOVIE_REVIEWS__]['unlabelled_data'] = unlabelled_data

	if (__TWITTER_CAMERON__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_CAMERON__], 'cameron')
		data[__TWITTER_CAMERON__]['train_data'] = train_data
		data[__TWITTER_CAMERON__]['train_labels'] = train_labels
		data[__TWITTER_CAMERON__]['unlabelled_data'] = unlabelled_data
		
	if (__TWITTER_CAMERON_2__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_CAMERON_2__], 'cameron-2')
		data[__TWITTER_CAMERON_2__]['train_data'] = train_data
		data[__TWITTER_CAMERON_2__]['train_labels'] = train_labels
		data[__TWITTER_CAMERON_2__]['unlabelled_data'] = unlabelled_data
	
	if (__TWITTER_DUGGAN__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_DUGGAN__], 'duggan')
		data[__TWITTER_DUGGAN__]['train_data'] = train_data
		data[__TWITTER_DUGGAN__]['train_labels'] = train_labels
		data[__TWITTER_DUGGAN__]['unlabelled_data'] = unlabelled_data
		
	if (__TWITTER_DUGGAN_MAIN__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_DUGGAN_MAIN__], 'duggan-main')
		data[__TWITTER_DUGGAN_MAIN__]['train_data'] = train_data
		data[__TWITTER_DUGGAN_MAIN__]['train_labels'] = train_labels
		data[__TWITTER_DUGGAN_MAIN__]['unlabelled_data'] = unlabelled_data
	
	if (__TWITTER_DUGGAN_VERDICT__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_DUGGAN_VERDICT__], 'duggan-verdict')
		data[__TWITTER_DUGGAN_VERDICT__]['train_data'] = train_data
		data[__TWITTER_DUGGAN_VERDICT__]['train_labels'] = train_labels
		data[__TWITTER_DUGGAN_VERDICT__]['unlabelled_data'] = unlabelled_data
		
	if (__TWITTER_IMMIGRATION_BILL__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_IMMIGRATION_BILL__], 'immigration-bill')
		data[__TWITTER_IMMIGRATION_BILL__]['train_data'] = train_data
		data[__TWITTER_IMMIGRATION_BILL__]['train_labels'] = train_labels
		data[__TWITTER_IMMIGRATION_BILL__]['unlabelled_data'] = unlabelled_data
		
	if (__TWITTER_IMMIGRATION_UK__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_IMMIGRATION_UK__], 'immigration-uk')
		data[__TWITTER_IMMIGRATION_UK__]['train_data'] = train_data
		data[__TWITTER_IMMIGRATION_UK__]['train_labels'] = train_labels
		data[__TWITTER_IMMIGRATION_UK__]['unlabelled_data'] = unlabelled_data
		
	if (__TWITTER_IMMIGRATION_EXT__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_IMMIGRATION_EXT__], 'immigration-ext')
		data[__TWITTER_IMMIGRATION_EXT__]['train_data'] = train_data
		data[__TWITTER_IMMIGRATION_EXT__]['train_labels'] = train_labels
		data[__TWITTER_IMMIGRATION_EXT__]['unlabelled_data'] = unlabelled_data
	
	if (__TWITTER_THEVOICEUK__ in keys):
		train_data, train_labels, unlabelled_data = fetch_twitter_fyp_dataset_vectorized(__DEFAULT_PATHS__[__TWITTER_THEVOICEUK__], 'thevoiceuk')
		data[__TWITTER_THEVOICEUK__]['train_data'] = train_data
		data[__TWITTER_THEVOICEUK__]['train_labels'] = train_labels
		data[__TWITTER_THEVOICEUK__]['unlabelled_data'] = unlabelled_data

	if (__TOY_EXAMPLE__ in keys):
		train_data, train_labels, test_data, test_labels, unlabelled_data = fetch_toy_example_dataset_vectorized(__DEFAULT_PATHS__[__TOY_EXAMPLE__])
		data[__TOY_EXAMPLE__]['train_data'] = train_data
		data[__TOY_EXAMPLE__]['train_labels'] = train_labels
		data[__TOY_EXAMPLE__]['test_data'] = test_data
		data[__TOY_EXAMPLE__]['test_labels'] = test_labels
		data[__TOY_EXAMPLE__]['unlabelled_data'] = unlabelled_data

	if (__WEBKB__ in keys):
		train_data, train_labels = fetch_webkb_dataset_vectorized(__DEFAULT_PATHS__[__WEBKB__])
		data[__WEBKB__]['train_data'] = train_data
		data[__WEBKB__]['train_labels'] = train_labels

	return data


def fetch_20newsgroups_dataset_binarised_and_vectorized(dataset_path, category_key, use_tfidf=False, return_raw=False,
														extraction_style='all', binarize=False, tf_normalisation=False,
														ngram_range=(1, 1),force_recreate_dataset=False):
	vectorized_labelled_train = None
	raw_train_data = None
	labels_train = None
	vectorized_labelled_test = None
	raw_test_data = None
	labels_test = None

	train_data_name = '%s_%s_vectors_labelled_train' % (category_key, 'tfidf' if use_tfidf else 'count')
	train_labels_name = '%s_labels_train' % (category_key,)
	test_data_name = '%s_%s_vectors_labelled_test' % (category_key, 'tfidf' if use_tfidf else 'count')
	test_labels_name = '%s_labels_test' % (category_key,)

	if (binarize):
		train_data_name += '_binary'
		test_data_name += '_binary'

	if (tf_normalisation):
		train_data_name += '_tf_norm'
		test_data_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		train_data_name = '_'.join([train_data_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		test_data_name = '_'.join([test_data_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	raw_train_data_name = '%s_raw_labelled_train' % (category_key,)
	raw_test_data_name = '%s_raw_labelled_test' % (category_key,)

	path_exists = os.path.exists(os.path.join(dataset_path, train_data_name)) if not return_raw else os.path.exists(os.path.join(dataset_path, raw_train_data_name))

	if (not force_recreate_dataset and path_exists):
		labels_train = joblib.load(os.path.join(dataset_path, train_labels_name))
		labels_test = joblib.load(os.path.join(dataset_path, test_labels_name))
		if (return_raw):
			raw_train_data = joblib.load(os.path.join(dataset_path, raw_train_data_name))
			raw_test_data = joblib.load(os.path.join(dataset_path, raw_test_data_name))
		else:
			vectorized_labelled_train = joblib.load(os.path.join(dataset_path, train_data_name))
			vectorized_labelled_test = joblib.load(os.path.join(dataset_path, test_data_name))
	else:
		prim_cat = category_key.split('_')[0]
		sec_cat = category_key.split('_')[1]

		''' Gunshot all
		dataset_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

		for i in xrange(len(dataset_categories) - 1):
			prim_cat = dataset_categories[i]
			for j in xrange(i + 1, len(dataset_categories)):
				sec_cat = dataset_categories[j]
		'''

		print('> CATEGORIES:', prim_cat, ' &', sec_cat)

		curr_tfidf_cat_key = '_'.join([prim_cat, sec_cat, 'tfidf'])
		curr_count_cat_key = '_'.join([prim_cat, sec_cat, 'count'])

		curr_tfidf_train_data_name = '%s_vectors_labelled_train' % (curr_tfidf_cat_key,)
		curr_tfidf_test_data_name = '%s_vectors_labelled_test' % (curr_tfidf_cat_key,)
		curr_count_train_data_name = '%s_vectors_labelled_train' % (curr_count_cat_key,)
		curr_count_test_data_name = '%s_vectors_labelled_test' % (curr_count_cat_key,)

		if (binarize):
			curr_tfidf_train_data_name += '_binary'
			curr_tfidf_test_data_name += '_binary'
			curr_count_train_data_name += '_binary'
			curr_count_test_data_name += '_binary'

		if (tf_normalisation):
			curr_count_train_data_name += '_tf_norm'
			curr_count_test_data_name += '_tf_norm'

		if (not min(ngram_range) == max(ngram_range) == 1):
			curr_count_train_data_name = '_'.join([curr_count_train_data_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
			curr_count_test_data_name = '_'.join([curr_count_test_data_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
			curr_tfidf_train_data_name = '_'.join([curr_tfidf_train_data_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
			curr_tfidf_test_data_name = '_'.join([curr_tfidf_test_data_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

		curr_test_labels_name = '%s_labels_test' % ('_'.join([prim_cat, sec_cat]),)
		curr_train_labels_name = '%s_labels_train' % ('_'.join([prim_cat, sec_cat]),)
		curr_count_vectorizer_name = 'count_vectorizer_%s' % ('_'.join([prim_cat, sec_cat]),)
		curr_tfidf_vectorizer_name = 'tfidf_vectorizer_%s' % ('_'.join([prim_cat, sec_cat]),)

		tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
		count_vectorizer = CountVectorizer(binary=binarize, ngram_range=ngram_range)
		transformer = TfidfTransformer(use_idf=False)

		train_bunch = fetch_20newsgroups(subset='train', categories=[prim_cat, sec_cat])
		test_bunch = fetch_20newsgroups(subset='test', categories=[prim_cat, sec_cat])

		tfidf_vectors_train = tfidf_vectorizer.fit_transform(train_bunch.data)
		tfidf_vectors_test = tfidf_vectorizer.transform(test_bunch.data)

		if (tf_normalisation):
			count_vectors_train = transformer.fit_transform(count_vectorizer.fit_transform(train_bunch.data))
			count_vectors_test = transformer.fit_transform(count_vectorizer.transform(test_bunch.data))
		else:
			count_vectors_train = count_vectorizer.fit_transform(train_bunch.data)
			count_vectors_test = count_vectorizer.transform(test_bunch.data)

		raw_train_data = train_bunch.data
		raw_test_data = test_bunch.data

		curr_labels_train = np.array(train_bunch.target)
		curr_labels_test = np.array(test_bunch.target)

		# Cache the stuff
		joblib.dump(tfidf_vectors_train, os.path.join(dataset_path, curr_tfidf_train_data_name))
		joblib.dump(count_vectors_train, os.path.join(dataset_path, curr_count_train_data_name))
		joblib.dump(tfidf_vectors_test, os.path.join(dataset_path, curr_tfidf_test_data_name))
		joblib.dump(count_vectors_test, os.path.join(dataset_path, curr_count_test_data_name))
		joblib.dump(curr_labels_train, os.path.join(dataset_path, curr_train_labels_name))
		joblib.dump(curr_labels_test, os.path.join(dataset_path, curr_test_labels_name))
		joblib.dump(count_vectorizer, os.path.join(dataset_path, curr_count_vectorizer_name))
		joblib.dump(tfidf_vectorizer, os.path.join(dataset_path, curr_tfidf_vectorizer_name))
		joblib.dump(raw_train_data, os.path.join(dataset_path, raw_train_data_name))
		joblib.dump(raw_test_data, os.path.join(dataset_path, raw_test_data_name))

		# Check if its the one we are looking for
		if (category_key == '_'.join([prim_cat, sec_cat]) and use_tfidf):
			vectorized_labelled_train = tfidf_vectors_train
			vectorized_labelled_test = tfidf_vectors_test
			labels_train = curr_labels_train
			labels_test = curr_labels_test
		elif (category_key == '_'.join([prim_cat, sec_cat]) and not use_tfidf):
			vectorized_labelled_train = count_vectors_train
			vectorized_labelled_test = count_vectors_test
			labels_train = curr_labels_train
			labels_test = curr_labels_test

	if (return_raw):
		return (raw_train_data, labels_train, raw_test_data, labels_test)
	return (vectorized_labelled_train, labels_train, vectorized_labelled_test, labels_test)


def fetch_20newsgroups_for_keys(dataset_path, category_list, use_tfidf=False, extraction_style='all', tf_normalisation=False, ngram_range=(1, 1)):
	tuple_list = list()
	for cat in category_list:
		tuple_list.append(fetch_20newsgroups_dataset_binarised_and_vectorized(dataset_path, cat, use_tfidf))

	return tuple_list


def fetch_20newsgroups_dataset_ternarised_and_vectorized(dataset_path, categories, use_tfidf=False, extraction_style='all', ngram_range=(1, 1)):
	pass


def fetch_20newsgroups_dataset_vectorized(dataset_path, use_tfidf=False, extraction_style='all',
										  binarize=False, tf_normalisation=False, ngram_range=(1, 1),
										  force_recreate_dataset=False):
	vectorized_labelled_train = None
	labels_train = None
	vectorized_labelled_test = None
	labels_test = None

	tfidf_vectors_train_name = 'tfidf_vectors_labelled_train' if not binarize else 'tfidf_vectors_labelled_train_binary'
	tfidf_vectors_test_name = 'tfidf_vectors_labelled_test' if not binarize else 'tfidf_vectors_labelled_test_binary'
	count_vectors_train_name = 'count_vectors_labelled_train' if not binarize else 'count_vectors_labelled_train_binary'
	count_vectors_test_name = 'count_vectors_labelled_test' if not binarize else 'count_vectors_labelled_test_binary'

	if (tf_normalisation):
		count_vectors_train_name += '_tf_norm'
		count_vectors_test_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		tfidf_vectors_train_name = '_'.join([tfidf_vectors_train_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_vectors_test_name = '_'.join([tfidf_vectors_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_vectors_train_name = '_'.join([count_vectors_train_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_vectors_test_name = '_'.join([count_vectors_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	if (not force_recreate_dataset and
			os.path.exists(os.path.join(dataset_path, tfidf_vectors_train_name if use_tfidf else count_vectors_test_name))):
		vectorized_labelled_train = joblib.load(os.path.join(dataset_path, tfidf_vectors_train_name if use_tfidf else count_vectors_train_name))
		vectorized_labelled_test = joblib.load(os.path.join(dataset_path, tfidf_vectors_test_name if use_tfidf else count_vectors_test_name))
		labels_train = joblib.load(os.path.join(dataset_path, 'train_labels'))
		labels_test = joblib.load(os.path.join(dataset_path, 'test_labels'))
	else:
		if (os.path.exists(dataset_path)):
			train_bunch = fetch_20newsgroups(subset='train')
			test_bunch = fetch_20newsgroups(subset='test')

			tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
			count_vectorizer = CountVectorizer(binary=binarize, ngram_range=ngram_range)
			transformer = TfidfTransformer(use_idf=False)

			tfidf_vectors_train = tfidf_vectorizer.fit_transform(train_bunch.data)
			tfidf_vectors_test = tfidf_vectorizer.transform(test_bunch.data)

			if (tf_normalisation):
				count_vectors_train = transformer.fit_transform(count_vectorizer.fit_transform(train_bunch.data))
				count_vectors_test = transformer.fit_transform(count_vectorizer.transform(test_bunch.data))
			else:
				count_vectors_train = count_vectorizer.fit_transform(train_bunch.data)
				count_vectors_test = count_vectorizer.transform(test_bunch.data)

			labels_train = np.array(train_bunch.target)
			labels_test = np.array(test_bunch.target)

			# Cache the stuff
			joblib.dump(tfidf_vectors_train, os.path.join(dataset_path, tfidf_vectors_train_name))
			joblib.dump(count_vectors_train, os.path.join(dataset_path, count_vectors_train_name))
			joblib.dump(tfidf_vectors_test, os.path.join(dataset_path, tfidf_vectors_test_name))
			joblib.dump(count_vectors_test, os.path.join(dataset_path, count_vectors_test_name))
			joblib.dump(labels_train, os.path.join(dataset_path, 'train_labels'))
			joblib.dump(labels_test, os.path.join(dataset_path, 'test_labels'))
			joblib.dump(count_vectorizer, os.path.join(dataset_path, 'count_vectorizer'))
			joblib.dump(tfidf_vectorizer, os.path.join(dataset_path, 'tfidf_vectorizer'))

			vectorized_labelled_train = tfidf_vectors_train if use_tfidf else count_vectors_train
			vectorized_labelled_test = tfidf_vectors_test if use_tfidf else count_vectors_test

	return (vectorized_labelled_train, labels_train, vectorized_labelled_test, labels_test)


def fetch_webkb_dataset_vectorized(dataset_path, use_tfidf=False, extraction_style='all',
								   binarize=False, tf_normalisation=False, ngram_range=(1, 1),
								   force_recreate_dataset=False):
	vectorized_train = None
	train_labels = None

	if (not force_recreate_dataset and
			os.path.exists(os.path.join(dataset_path, 'tfidf_vectors_labelled' if use_tfidf else 'count_vectors_labelled'))):
		vectorized_train = joblib.load(os.path.join(dataset_path, 'tfidf_vectors_labelled' if use_tfidf else 'count_vectors_labelled'))
		train_labels = joblib.load(os.path.join(dataset_path, 'train_labels'))
	else:
		tfidf_vectorizer = TfidfVectorizer(decode_error='replace')
		count_vectorizer = CountVectorizer(decode_error='replace', binary=binarize)
		transformer = TfidfTransformer(use_idf=False)
		'''
		for path in glob.glob(os.path.join(dataset_path, '*')):
			_, label = os.path.split(path)
			for sub_path in glob.glob(os.path.join(path, '*')):
				for file_path in glob.glob(os.path.join(sub_path, '*')):
					soup = BeautifulSoup(str(open(file_path).readlines()))
					content = soup.get_text(' ', strip=True).replace(r'\r', '').replace(r'\n', '')
					#TODO: When time, get rid of the headers....
					print('CONTENT:', content)
		'''
	return (vectorized_train, train_labels)


def fetch_toy_example_dataset_vectorized(dataset_path, use_tfidf=False, wrap_in_list=False,
										 extraction_style='all', tf_normalisation=False, ngram_range=(1, 1),
										 force_recreate_dataset=False):
	vectorized_labelled_train = None
	vectorized_labelled_test = None
	vectorized_unlabelled = None
	train_labels = None
	test_labels = None

	if (not force_recreate_dataset and
			os.path.exists(os.path.join(dataset_path, 'tfidf_vectors_labelled_train' if use_tfidf else 'count_vectors_labelled_train'))):
		vectorized_labelled_train = joblib.load(os.path.join(dataset_path, 'tfidf_vectors_labelled_train' if use_tfidf else 'count_vectors_labelled_train'))
		vectorized_labelled_test = joblib.load(os.path.join(dataset_path, 'tfidf_vectors_labelled_test' if use_tfidf else 'count_vectors_labelled_test'))
		vectorized_unlabelled = joblib.load(os.path.join(dataset_path, 'tfidf_vectors_unlabelled' if use_tfidf else 'count_vectors_unlabelled'))
		train_labels = joblib.load(os.path.join(dataset_path, 'labels_train'))
		test_labels = joblib.load(os.path.join(dataset_path, 'labels_test'))
	else:
		if (os.path.exists(dataset_path)):
			tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
			count_vectorizer = CountVectorizer(ngram_range=ngram_range)
			transformer = TfidfTransformer(use_idf=False)

			labelled_train = [d.strip() for d in open(os.path.join(dataset_path, 'labelled_train_docs.txt')).readlines()]
			labelled_test = [d.strip() for d in open(os.path.join(dataset_path, 'labelled_test_docs.txt')).readlines()]
			unlabelled = [d.strip() for d in open(os.path.join(dataset_path, 'unlabelled_docs.txt')).readlines()]
			train_labels = np.array([l.strip() for l in open(os.path.join(dataset_path, 'train_labels.txt')).readlines()])
			test_labels = np.array([l.strip() for l in open(os.path.join(dataset_path, 'test_labels.txt')).readlines()])

			tfidf_vectors_train = tfidf_vectorizer.fit_transform(labelled_train)
			tfidf_vectors_test = tfidf_vectorizer.transform(labelled_test)
			tfidf_vectors_unlabelled = tfidf_vectorizer.transform(unlabelled)

			count_vectors_train = count_vectorizer.fit_transform(labelled_train)
			count_vectors_test = count_vectorizer.transform(labelled_test)
			count_vectors_unlabelled = count_vectorizer.transform(unlabelled)

			# Cache the stuff
			joblib.dump(tfidf_vectors_train, os.path.join(dataset_path, 'tfidf_vectors_labelled_train'))
			joblib.dump(tfidf_vectors_test, os.path.join(dataset_path, 'tfidf_vectors_labelled_test'))
			joblib.dump(tfidf_vectors_unlabelled, os.path.join(dataset_path, 'tfidf_vectors_unlabelled'))

			joblib.dump(count_vectors_train, os.path.join(dataset_path, 'count_vectors_labelled_train'))
			joblib.dump(count_vectors_test, os.path.join(dataset_path, 'count_vectors_labelled_test'))
			joblib.dump(count_vectors_unlabelled, os.path.join(dataset_path, 'count_vectors_unlabelled'))

			joblib.dump(train_labels, os.path.join(dataset_path, 'labels_train'))
			joblib.dump(test_labels, os.path.join(dataset_path, 'labels_test'))

			joblib.dump(tfidf_vectorizer, os.path.join(dataset_path, 'tfidf_vectorizer'))
			joblib.dump(count_vectorizer, os.path.join(dataset_path, 'count_vectorizer'))

	return (vectorized_labelled_train, train_labels, vectorized_labelled_test, test_labels, vectorized_unlabelled) if not wrap_in_list else [(vectorized_labelled_train, train_labels, vectorized_labelled_test, test_labels, vectorized_unlabelled)]


def fetch_movie_reviews_dataset_vectorized(dataset_path=os.path.join(paths.get_dataset_path(), 'movie_reviews', 'aclImdb'),
										   use_tfidf=False, wrap_in_list=False, count_dtype=np.int64,
										   tfidf_dtype=np.float64, return_raw=False, extraction_style='all',
										   binarize=False, tf_normalisation=False, ngram_range=(1, 1),
										   force_recreate_dataset=False):
	vectorized_labelled_train = None
	vectorized_labelled_test = None
	vectorized_unlabelled = None
	train_labels = None
	test_labels = None

	tfidf_vectorizer = TfidfVectorizer(decode_error='replace', dtype=tfidf_dtype, ngram_range=ngram_range)
	count_vectorizer = CountVectorizer(decode_error='replace', dtype=count_dtype, binary=binarize, ngram_range=ngram_range)
	transformer = TfidfTransformer(use_idf=False)

	tfidf_labelled_name_train = 'tfidf_vectors_labelled_train_' + extraction_style if extraction_style != None else 'tfidf_vectors_labelled_train'
	tfidf_labelled_name_test = 'tfidf_vectors_labelled_test_' + extraction_style if extraction_style != None else 'tfidf_vectors_labelled_test'
	tfidf_unlabelled_name = 'tfidf_vectors_unlabelled_' + extraction_style if extraction_style != None else 'tfidf_vectors_unlabelled'
	count_labelled_name_train = 'count_vectors_labelled_train_' + extraction_style if extraction_style != None else 'count_vectors_labelled_train'
	count_labelled_name_test = 'count_vectors_labelled_test_' + extraction_style if extraction_style != None else 'count_vectors_labelled_test'
	count_unlabelled_name = 'count_vectors_unlabelled_' + extraction_style if extraction_style != None else 'count_vectors_unlabelled'

	if (binarize):
		tfidf_labelled_name_train += '_binary'
		tfidf_labelled_name_test += '_binary'
		tfidf_unlabelled_name += '_binary'
		count_labelled_name_train += '_binary'
		count_labelled_name_test += '_binary'
		count_unlabelled_name += '_binary'

	if (tf_normalisation):
		count_labelled_name_train += '_tf_norm'
		count_labelled_name_test += '_tf_norm'
		count_unlabelled_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		tfidf_labelled_name_train = '_'.join([tfidf_labelled_name_train, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_labelled_name_test = '_'.join([tfidf_labelled_name_test, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_unlabelled_name = '_'.join([tfidf_unlabelled_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_labelled_name_train = '_'.join([count_labelled_name_train, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_labelled_name_test = '_'.join([count_labelled_name_test, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_unlabelled_name = '_'.join([count_unlabelled_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	if (not force_recreate_dataset and
			os.path.exists(os.path.join(dataset_path,tfidf_labelled_name_train if use_tfidf else count_labelled_name_train)) and os.path.exists(os.path.join(dataset_path, 'train_docs'))):
		vectorized_labelled_train = joblib.load(os.path.join(dataset_path, tfidf_labelled_name_train if use_tfidf else count_labelled_name_train))
		vectorized_labelled_test = joblib.load(os.path.join(dataset_path, tfidf_labelled_name_test if use_tfidf else count_labelled_name_test))
		train_labels = joblib.load(os.path.join(dataset_path, 'labels_train'))
		test_labels = joblib.load(os.path.join(dataset_path, 'labels_test'))
		vectorized_unlabelled = joblib.load(os.path.join(dataset_path, tfidf_unlabelled_name if use_tfidf else count_unlabelled_name))
		train_docs = joblib.load(os.path.join(dataset_path, 'train_docs'))
		test_docs = joblib.load(os.path.join(dataset_path, 'test_docs'))
		unlabelled_docs = joblib.load(os.path.join(dataset_path, 'unlabelled_docs'))
	else:
		if (os.path.exists(dataset_path)):
			# Read training data
			train_path = os.path.join(dataset_path, 'train')
			test_path = os.path.join(dataset_path, 'test')
			unlabelled_path = os.path.join(dataset_path, 'train', 'unsup')
			target_names = ['pos', 'neg']
			label_map = dict((value, key) for key, value in dict(enumerate(target_names)).items())
			train_targets = list()
			test_targets = list()
			train_docs = list()
			test_docs = list()
			unlabelled_docs = list()

			# Read labelled data
			for p, X, y in zip([train_path, test_path], [train_docs, test_docs], [train_targets, test_targets]):
				for t in target_names:
					os.chdir(os.path.join(p, t))
					for f in glob.glob('*.txt'):
						with open(f) as curr_file:
							y.append(label_map[t])
							X.append(curr_file.read())

			# Read unlabelled data
			os.chdir(unlabelled_path)
			for f in glob.glob('*.txt'):
				with open(f) as curr_file:
					unlabelled_docs.append(curr_file.read())

			if (extraction_style == 'all'):
				all_data = train_docs + unlabelled_docs
				tfidf_vectorizer.fit(all_data)
				count_vectorizer.fit(all_data)
			else:
				tfidf_vectorizer.fit(train_docs)
				count_vectorizer.fit(train_docs)

			tfidf_vectors_labelled_train = tfidf_vectorizer.transform(train_docs)
			tfidf_vectors_labelled_test = tfidf_vectorizer.transform(test_docs)
			tfidf_vectors_unlabelled = tfidf_vectorizer.transform(unlabelled_docs)

			if (tf_normalisation):
				count_vectors_labelled_test = transformer.fit_transform(count_vectorizer.transform(test_docs))
				count_vectors_labelled_train = transformer.fit_transform(count_vectorizer.transform(train_docs))
				count_vectors_unlabelled = transformer.fit_transform(count_vectorizer.transform(unlabelled_docs))
			else:
				count_vectors_labelled_test = count_vectorizer.transform(test_docs)
				count_vectors_labelled_train = count_vectorizer.transform(train_docs)
				count_vectors_unlabelled = count_vectorizer.transform(unlabelled_docs)

			train_labels = np.array(train_targets)
			test_labels = np.array(test_targets)

			# Cache the stuff
			joblib.dump(tfidf_vectors_labelled_train, os.path.join(dataset_path, tfidf_labelled_name_train))
			joblib.dump(count_vectors_labelled_train, os.path.join(dataset_path, count_labelled_name_train))
			joblib.dump(train_labels, os.path.join(dataset_path, 'labels_train'))
			joblib.dump(tfidf_vectors_labelled_test, os.path.join(dataset_path, tfidf_labelled_name_test))
			joblib.dump(count_vectors_labelled_test, os.path.join(dataset_path, count_labelled_name_test))
			joblib.dump(test_labels, os.path.join(dataset_path, 'labels_test'))
			joblib.dump(tfidf_vectors_unlabelled, os.path.join(dataset_path, tfidf_unlabelled_name))
			joblib.dump(count_vectors_unlabelled, os.path.join(dataset_path, count_unlabelled_name))
			joblib.dump(tfidf_vectorizer, os.path.join(dataset_path, 'tfidf_vectorizer'))
			joblib.dump(count_vectorizer, os.path.join(dataset_path, 'count_vectorizer'))
			#pickle.dump(train_docs, open(os.path.join(dataset_path, 'train_docs'), 'w'))
			#pickle.dump(test_docs, open(os.path.join(dataset_path, 'test_docs'), 'w'))
			#pickle.dump(unlabelled_docs, open(os.path.join(dataset_path, 'unlabelled_docs'), 'w'))
			joblib.dump(train_docs, os.path.join(dataset_path, 'train_docs'))
			joblib.dump(test_docs, os.path.join(dataset_path, 'test_docs'))
			joblib.dump(unlabelled_docs, os.path.join(dataset_path, 'unlabelled_docs'))

			vectorized_labelled_train = tfidf_vectors_labelled_train if use_tfidf else count_vectors_labelled_train
			vectorized_labelled_test = tfidf_vectors_labelled_test if use_tfidf else count_vectors_labelled_test
			vectorized_unlabelled = tfidf_vectors_unlabelled if use_tfidf else count_vectors_unlabelled

	if (return_raw):
		return (train_docs, train_labels, test_docs, test_labels, unlabelled_docs)
	return (vectorized_labelled_train, train_labels, vectorized_labelled_test, test_labels, vectorized_unlabelled) if not wrap_in_list else [(vectorized_labelled_train, train_labels, vectorized_labelled_test, test_labels, vectorized_unlabelled)]


def fetch_sms_spam_collection_dataset_vectorized(dataset_path, use_tfidf=False, wrap_in_list=False, return_raw=False,
												 extraction_style='all', binarize=False, tf_normalisation=False,
												 ngram_range=(1, 1), force_recreate_dataset=False):
	vectorized_labelled = None
	labels = None

	tfidf_vectorizer = TfidfVectorizer(decode_error='replace', ngram_range=ngram_range)
	count_vectorizer = CountVectorizer(decode_error='replace', binary=binarize, ngram_range=ngram_range)

	tfidf_vectors_name = 'tfidf_vectors' if not binarize else 'tfidf_vectors_binary'
	count_vectors_name = 'count_vectors' if not binarize else 'count_vectors_binary'

	if (tf_normalisation):
		count_vectors_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		tfidf_vectors_name = '_'.join([tfidf_vectors_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_vectors_name = '_'.join([count_vectors_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	# Check if cache exists
	tail, _ = os.path.split(dataset_path)
	cache_path = os.path.join(tail, tfidf_vectors_name if use_tfidf else count_vectors_name)
	if (not force_recreate_dataset and
			os.path.exists(cache_path) and os.path.exists(os.path.join(tail, 'raw_data'))):
		vectorized_labelled = joblib.load(cache_path)
		labels = joblib.load(os.path.join(tail, 'labels'))
		raw_data = joblib.load(os.path.join(tail, 'raw_data'))
	else:
		if (os.path.exists(dataset_path)):
			label_dist = collections.defaultdict(int)
			target_names = ['ham', 'spam']
			label_map = dict((value, key) for key, value in dict(enumerate(target_names)).items())
			docs = list()
			targets = list()
			with open(dataset_path, 'r') as raw_dataset:
				for line in raw_dataset:
					comps = line.split('\t')
					l = comps[0].strip()
					targets.append(label_map[l])
					docs.append(comps[1].strip())
					label_dist[l] += 1

			transformer = TfidfTransformer(use_idf=False)

			tfidf_labelled = tfidf_vectorizer.fit_transform(docs)

			if (tf_normalisation):
				count_labelled = transformer.fit_transform(count_vectorizer.fit_transform(docs))
			else:
				count_labelled = count_vectorizer.fit_transform(docs)

			labels = np.array(targets)
			raw_data = docs

			# Cache the stuff
			joblib.dump(tfidf_labelled, os.path.join(tail, tfidf_vectors_name))
			joblib.dump(count_labelled, os.path.join(tail, count_vectors_name))
			joblib.dump(labels, os.path.join(tail, 'labels'))
			json.dumps(label_dist, os.path.join(tail, 'label_distribution.json'))
			joblib.dump(tfidf_vectorizer, os.path.join(tail, 'tfidf_vectorizer'))
			joblib.dump(count_vectorizer, os.path.join(tail, 'count_vectorizer'))
			#pickle.dump(docs, open(os.path.join(dataset_path, 'docs'), 'w'))
			joblib.dump(docs, os.path.join(tail, 'raw_data'))

			vectorized_labelled = tfidf_labelled if use_tfidf else count_labelled

	if (return_raw):
		return raw_data, labels
	return (vectorized_labelled, labels) if not wrap_in_list else [(vectorized_labelled, labels)]


def fetch_aptemod_dataset_vectorized(dataset_path, use_tfidf=False, wrap_in_list=False, return_raw=False, top_n_classes=5,
									 extraction_style='all', binarize=False, tf_normalisation=False, ngram_range=(1, 1),
									 force_recreate_dataset=False):
	vectorized_labelled_train  = None
	train_labels = None
	vectorized_labelled_test = None
	test_labels = None
	vectorized_unlabelled = None
	raw_labelled_train = None
	raw_labelled_test = None

	tfidf_vectorizer = TfidfVectorizer(decode_error='replace', ngram_range=ngram_range)
	count_vectorizer = CountVectorizer(decode_error='replace', binary=binarize, ngram_range=ngram_range)

	tfidf_vectors_train_name = 'tfidf_vectors_labelled_train' if not binarize else 'tfidf_vectors_labelled_train_binary'
	tfidf_vectors_test_name = 'tfidf_vectors_labelled_test' if not binarize else 'tfidf_vectors_labelled_test_binary'
	count_vectors_train_name = 'count_vectors_labelled_train' if not binarize else 'count_vectors_labelled_train_binary'
	count_vectors_test_name = 'count_vectors_labelled_test' if not binarize else 'count_vectors_labelled_test_binary'

	if (tf_normalisation):
		count_vectors_train_name += '_tf_norm'
		count_vectors_test_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		tfidf_vectors_train_name = '_'.join([tfidf_vectors_train_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_vectors_test_name = '_'.join([tfidf_vectors_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_vectors_train_name = '_'.join([count_vectors_train_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_vectors_test_name = '_'.join([count_vectors_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	# Check if cache exists
	cache_path = os.path.join(dataset_path, '%d_%s' % (top_n_classes, tfidf_vectors_train_name if use_tfidf else count_vectors_train_name))
	if (not force_recreate_dataset and
			os.path.exists(cache_path) and os.path.exists(os.path.join(dataset_path, 'raw_labelled_train'))):
		vectorized_labelled_train = joblib.load(os.path.join(dataset_path, tfidf_vectors_train_name if use_tfidf else count_vectors_train_name))
		vectorized_labelled_test = joblib.load(os.path.join(dataset_path, tfidf_vectors_test_name if use_tfidf else count_vectors_test_name))
		vectorized_unlabelled = joblib.load(os.path.join(dataset_path, 'tfidf_vectors_unlabelled' if use_tfidf else 'count_vectors_unlabelled'))
		train_labels = joblib.load(os.path.join(dataset_path, 'labels_train'))
		test_labels = joblib.load(os.path.join(dataset_path, 'labels_test'))
		raw_labelled_train = joblib.load(os.path.join(dataset_path, 'raw_labelled_train'))
		raw_labelled_test = joblib.load(os.path.join(dataset_path, 'raw_labelled_test'))
	else:
		if (os.path.exists(dataset_path)):
			aptemod_index = AptemodIndex(dataset_path)
			aptemod_index.load_index()

			# Hardcoded: fetch 10 largest categories, filter overlapping docs and docs not belonging to any of the target classes as in Lucas, Downey (2013)
			target_names = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']
			target_names = target_names[:top_n_classes]
			label_map = dict((value, key) for key, value in dict(enumerate(target_names)).items())

			# Training Data
			train_corpus = open(os.path.join(dataset_path, 'aptemod_train.tsv'))
			train_docs = list()
			train_targets = list()
			for line in train_corpus:
				comps = line.split('\t')
				doc_id = comps[0]
				doc_text = comps[1]

				# Get set of possible targets, if set intersection=0 -> filter, if set intersection > 1 -> filter
				doc_targets = set(aptemod_index.topic_list_for_doc_id(doc_id))
				if (len(doc_targets & set(target_names)) == 1):
					train_docs.append(doc_text)

					# Find label
					for t in doc_targets:
						if t in label_map:
							train_targets.append(label_map[t])
							break

			train_corpus.close()

			# Testing Data
			test_corpus = open(os.path.join(dataset_path, 'aptemod_test.tsv'))
			test_docs = list()
			test_targets = list()
			for line in test_corpus:
				comps = line.split('\t')
				doc_id = comps[0]
				doc_text = comps[1]

				# Get set of possible targets, if set intersection=0 -> filter, if set intersection > 1 -> filter
				doc_targets = set(aptemod_index.topic_list_for_doc_id(doc_id))
				if (len(doc_targets & set(target_names)) == 1):
					test_docs.append(doc_text)

					# Find label
					for t in doc_targets:
						if t in label_map:
							test_targets.append(label_map[t])
							break

			# Unlabelled Data
			unlabelled_corpus = open(os.path.join(dataset_path, 'aptemod_unlabelled.tsv'))
			unlabelled_docs = list()
			for line in unlabelled_corpus:
				unlabelled_docs.append(line.split('\t')[1])

			transformer = TfidfTransformer(use_idf=False)

			# Transform
			tfidf_vectors_labelled_train = tfidf_vectorizer.fit_transform(train_docs)
			tfidf_vectors_labelled_test = tfidf_vectorizer.transform(test_docs)
			tfidf_vectors_unlabelled = tfidf_vectorizer.transform(unlabelled_docs)

			if (tf_normalisation):
				count_vectors_labelled_train = transformer.fit_transform(count_vectorizer.fit_transform(train_docs))
				count_vectors_labelled_test = transformer.fit_transform(count_vectorizer.transform(test_docs))
				count_vectors_unlabelled = transformer.fit_transform(count_vectorizer.transform(unlabelled_docs))
			else:
				count_vectors_labelled_train = count_vectorizer.fit_transform(train_docs)
				count_vectors_labelled_test = count_vectorizer.transform(test_docs)
				count_vectors_unlabelled = count_vectorizer.transform(unlabelled_docs)

			train_labels = np.array(train_targets)
			test_labels = np.array(test_targets)

			# Cache
			joblib.dump(tfidf_vectors_labelled_train, os.path.join(dataset_path, tfidf_vectors_train_name))
			joblib.dump(tfidf_vectors_labelled_test, os.path.join(dataset_path, tfidf_vectors_test_name))
			joblib.dump(tfidf_vectors_unlabelled, os.path.join(dataset_path, 'tfidf_vectors_unlabelled'))
			joblib.dump(count_vectors_labelled_train, os.path.join(dataset_path, count_vectors_train_name))
			joblib.dump(count_vectors_labelled_test, os.path.join(dataset_path, count_vectors_test_name))
			joblib.dump(count_vectors_unlabelled, os.path.join(dataset_path, 'count_vectors_unlabelled'))
			joblib.dump(train_labels, os.path.join(dataset_path, 'labels_train'))
			joblib.dump(test_labels, os.path.join(dataset_path, 'labels_test'))
			joblib.dump(tfidf_vectorizer, os.path.join(dataset_path, 'tfidf_vectorizer'))
			joblib.dump(count_vectorizer, os.path.join(dataset_path, 'count_vectorizer'))
			#pickle.dump(train_docs, open(os.path.join(dataset_path, 'train_docs'), 'w'))
			#pickle.dump(test_docs, open(os.path.join(dataset_path, 'test_docs'), 'w'))
			#pickle.dump(unlabelled_docs, open(os.path.join(dataset_path, 'unlabelled_docs'), 'w'))
			joblib.dump(train_docs, os.path.join(dataset_path, 'raw_labelled_train'))
			joblib.dump(test_docs, os.path.join(dataset_path, 'raw_labelled_test'))
			joblib.dump(unlabelled_docs, os.path.join(dataset_path, 'raw_unlabelled'))

			vectorized_labelled_train = tfidf_vectors_labelled_train if use_tfidf else count_vectors_labelled_train
			vectorized_labelled_test = tfidf_vectors_labelled_test if use_tfidf else count_vectors_labelled_test
			vectorized_unlabelled = tfidf_vectors_unlabelled if use_tfidf else count_vectors_unlabelled

	if (return_raw):
		return (raw_labelled_train, train_labels, raw_labelled_test, test_labels)
	return (vectorized_labelled_train, train_labels, vectorized_labelled_test, test_labels) if not wrap_in_list else [(vectorized_labelled_train, train_labels, vectorized_labelled_test, test_labels)]


def fetch_rcv1_dataset_vectorized(dataset_path, use_tfidf=False, wrap_in_list=False, return_raw=False,
								  extraction_style='all', binarize=False, tf_normalisation=False,
								  ngram_range=(1, 1), force_recreate_dataset=False):
	vectorized_labelled = None
	labels = None
	raw_labelled = None

	tfidf_vectorizer = TfidfVectorizer(decode_error='replace', tokenizer=lambda l: l.split(), ngram_range=ngram_range)
	count_vectorizer = CountVectorizer(decode_error='replace', tokenizer=lambda l: l.split(), binary=binarize, ngram_range=ngram_range)

	tfidf_vectors_name = 'tfidf_vectors' if not binarize else 'tfidf_vectors_binary'
	count_vectors_name = 'count_vectors' if not binarize else 'count_vectors_binary'

	if (tf_normalisation):
		count_vectors_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		tfidf_vectors_name = '_'.join([tfidf_vectors_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_vectors_name = '_'.join([count_vectors_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	# Check if cache exists
	cache_path = os.path.join(dataset_path, tfidf_vectors_name if use_tfidf else count_vectors_name)
	path_exists = os.path.exists(cache_path) if not return_raw else os.path.exists(os.path.join(dataset_path, 'docs'))
	if (not force_recreate_dataset and path_exists):
		labels = joblib.load(os.path.join(dataset_path, 'labels'))
		if (not return_raw):
			vectorized_labelled = joblib.load(cache_path)
			raw_labelled = []
		else:
			vectorized_labelled = np.array([])
			raw_labelled = joblib.load(os.path.join(dataset_path, 'docs'))
	else:
		if (os.path.exists(dataset_path)):
			tail, _ = os.path.split(dataset_path)
			rcv1_index = RCV1Index(os.path.join(tail, 'rcv1'))
			rcv1_index.load_index()

			# Hardcoded: fetch 5 largest categories, filter overlapping docs and docs not belonging to any of the target classes as in Lucas, Downey (2013)
			target_names = ['CCAT', 'GCAT', 'MCAT', 'ECAT', 'GPOL']
			label_map = dict((value, key) for key, value in dict(enumerate(target_names)).items())
			docs = list()
			targets = list()

			# TODO: Build label map from all categories, then select documents and vectorize, then get on with other datasets and EM papers
			archive_path = dataset_path if dataset_path.endswith('RCV1.txt.bz2') else os.path.join(dataset_path, 'RCV1.txt.bz2')

			with bz2.BZ2File(archive_path, 'r') as rcv1_compressed:
				for line in rcv1_compressed:

					comps = line.split('\t') # 0=doc_id, 1=mattis_label, 2=timestamp, 3=document
					doc_id = comps[0]

					# Get set of possible targets, if set intersection=0 -> filter, if set intersection > 1 -> filter
					doc_targets = set(rcv1_index.topic_list_for_doc_id(doc_id))
					if (len(doc_targets & set(target_names)) == 1):
						docs.append(comps[3])

						# Find label
						for t in doc_targets:
							if t in label_map:
								targets.append(label_map[t])
								break

			transformer = TfidfTransformer(use_idf=False)

			tfidf_labelled = tfidf_vectorizer.fit_transform(docs)

			if (tf_normalisation):
				count_labelled = transformer.fit_transform(count_vectorizer.fit_transform(docs))
			else:
				count_labelled = count_vectorizer.fit_transform(docs)

			labels = np.array(targets)

			# Cache the stuff
			cache_path = os.path.join(tail, 'rcv1')

			raw_labelled = docs

			joblib.dump(tfidf_labelled, os.path.join(cache_path, tfidf_vectors_name))
			joblib.dump(count_labelled, os.path.join(cache_path, count_vectors_name))
			joblib.dump(labels, os.path.join(cache_path, 'labels'))
			#joblib.dump(tfidf_vectorizer, os.path.join(dataset_path, 'tfidf_vectorizer'))
			#joblib.dump(count_vectorizer, os.path.join(dataset_path, 'count_vectorizer'))
			#pickle.dump(docs, open(os.path.join(dataset_path, 'docs'), 'w'))
			joblib.dump(docs, os.path.join(dataset_path, 'docs'))

			vectorized_labelled = tfidf_labelled if use_tfidf else count_labelled

	if (return_raw):
		return (raw_labelled, labels)
	return (vectorized_labelled, labels) if not wrap_in_list else [(vectorized_labelled, labels)]


def fetch_ws_paper_dataset_vectorized(dataset_path, dataset_name, use_tfidf=False, extraction_style='all',
									  binarize=False, tf_normalisation=False, ngram_range=(1, 1),
									  force_recreate_dataset=False, load_vectorizer=False, min_df=None):
	vectorized_labelled_train = None
	train_labels = None
	vectorized_labelled_test = None
	test_labels = None
	vectorized_unlabelled = None
	label_map = None
	labelled_features = None
	vocab = None
	count_vectorizer = None
	tfidf_vectorizer = None

	tfidf_labelled_train_name = 'tfidf_vectors_labelled_train_' + extraction_style if extraction_style != None else 'tfidf_vectors_labelled_train'
	tfidf_labelled_test_name = 'tfidf_vectors_labelled_test_' + extraction_style if extraction_style != None else 'tfidf_vectors_labelled_test'
	tfidf_unlabelled_name = 'tfidf_vectors_unlabelled_' + extraction_style if extraction_style != None else 'tfidf_vectors_unlabelled'
	tfidf_vectorizer_name = 'tfidf_vectorizer_' + extraction_style if extraction_style != None else 'tfidf_vectorizer'
	count_labelled_train_name = 'count_vectors_labelled_train_' + extraction_style if extraction_style != None else 'count_vectors_labelled_train'
	count_labelled_test_name = 'count_vectors_labelled_test_' + extraction_style if extraction_style != None else 'count_vectors_labelled_test'
	count_unlabelled_test_name = 'count_vectors_unlabelled_' + extraction_style if extraction_style != None else 'count_vectors_unlabelled'
	count_vectorizer_name = 'count_vectorizer_' + extraction_style if extraction_style != None else 'count_vectorizer'

	if (binarize):
		tfidf_labelled_train_name += '_binary'
		tfidf_labelled_test_name += '_binary'
		tfidf_unlabelled_name += '_binary'
		count_labelled_train_name += '_binary'
		count_labelled_test_name += '_binary'
		count_unlabelled_test_name += '_binary'
		
	if (tf_normalisation):
		count_labelled_train_name += '_tf_norm'
		count_labelled_test_name += '_tf_norm'
		count_unlabelled_test_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		tfidf_labelled_train_name = '_'.join([tfidf_labelled_train_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_labelled_test_name = '_'.join([tfidf_labelled_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_unlabelled_name = '_'.join([tfidf_unlabelled_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_vectorizer_name = '_'.join([tfidf_vectorizer_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_labelled_train_name = '_'.join([count_labelled_train_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_labelled_test_name = '_'.join([count_labelled_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_unlabelled_test_name = '_'.join([count_unlabelled_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_vectorizer_name = '_'.join([count_vectorizer_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	# Vocab Name
	vocab_name = '_'.join(['vocab', 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	# Check if cached stuff exists
	if (not force_recreate_dataset and
			os.path.exists(os.path.join(dataset_path, dataset_name, tfidf_labelled_train_name if use_tfidf else count_labelled_train_name))):
		vectorized_labelled_train = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_labelled_train_name if use_tfidf else count_labelled_train_name))
		train_labels = joblib.load(os.path.join(dataset_path, dataset_name, 'train_labels'))
		vectorized_labelled_test = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_labelled_test_name if use_tfidf else count_labelled_test_name))
		test_labels = joblib.load(os.path.join(dataset_path, dataset_name, 'test_labels'))
		vectorized_unlabelled = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_unlabelled_name if use_tfidf else count_unlabelled_test_name))
		label_map = joblib.load(os.path.join(dataset_path, dataset_name, 'label_map'))
		labelled_features = joblib.load(os.path.join(dataset_path, dataset_name, 'labelled_features'))

		if (os.path.exists(os.path.join(dataset_path, dataset_name, vocab_name))):
			vocab = joblib.load(os.path.join(dataset_path, dataset_name, vocab_name))
		else:
			vocab = joblib.load(os.path.join(dataset_path, dataset_name, 'vocab'))

		if (load_vectorizer):
			tfidf_vectorizer = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_vectorizer_name))
			count_vectorizer = joblib.load(os.path.join(dataset_path, dataset_name, count_vectorizer_name))
		else:
			tfidf_vectorizer = None
			count_vectorizer = None
	else:
		if (os.path.exists(os.path.join(dataset_path, dataset_name))):
			raw_labelled_train = joblib.load(os.path.join(dataset_path, dataset_name, 'raw_training_docs'))
			raw_train_labels = joblib.load(os.path.join(dataset_path, dataset_name, 'raw_training_labels'))
			raw_labelled_test = joblib.load(os.path.join(dataset_path, dataset_name, 'raw_gold_standard_docs'))
			raw_test_labels = joblib.load(os.path.join(dataset_path, dataset_name, 'raw_gold_standard_labels'))
			raw_unlabelled = joblib.load(os.path.join(dataset_path, dataset_name, 'raw_unlabelled_docs'))
			vocab = joblib.load(os.path.join(dataset_path, dataset_name, 'vocab'))
			labelled_features = joblib.load(os.path.join(dataset_path, dataset_name, 'labelled_features'))
			label_map = joblib.load(os.path.join(dataset_path, dataset_name, 'label_map'))

			tfidf_vectorizer = TfidfVectorizer(decode_error='replace', ngram_range=ngram_range)
			count_vectorizer = CountVectorizer(decode_error='replace', binary=binarize, ngram_range=ngram_range)
			transformer = TfidfTransformer(use_idf=False)

			if (extraction_style == 'all'):
				all_data = raw_labelled_train + raw_unlabelled
				tfidf_vectorizer.fit(all_data)
				count_vectorizer.fit(all_data)
			else:
				tfidf_vectorizer.fit(raw_labelled_train)
				count_vectorizer.fit(raw_labelled_train)

			tfidf_vectors_labelled_train = tfidf_vectorizer.transform(raw_labelled_train)
			tfidf_vectors_labelled_test = tfidf_vectorizer.transform(raw_labelled_test)
			tfidf_vectors_unlabelled = tfidf_vectorizer.transform(raw_unlabelled)

			if (tf_normalisation):
				count_vectors_labelled_train = transformer.fit_transform(count_vectorizer.transform(raw_labelled_train))
				count_vectors_labelled_test = transformer.fit_transform(count_vectorizer.transform(raw_labelled_test))
				count_vectors_unlabelled = transformer.fit_transform(count_vectorizer.transform(raw_unlabelled))
			else:
				count_vectors_labelled_train = count_vectorizer.transform(raw_labelled_train)
				count_vectors_labelled_test = count_vectorizer.transform(raw_labelled_test)
				count_vectors_unlabelled = count_vectorizer.transform(raw_unlabelled)


			train_labels = np.array(raw_train_labels)
			test_labels = np.array(raw_test_labels)

			# Cache the stuff
			joblib.dump(tfidf_vectors_labelled_train, os.path.join(dataset_path, dataset_name, tfidf_labelled_train_name))
			joblib.dump(tfidf_vectors_labelled_test, os.path.join(dataset_path, dataset_name, tfidf_labelled_test_name))
			joblib.dump(tfidf_vectors_unlabelled, os.path.join(dataset_path, dataset_name, tfidf_unlabelled_name))

			joblib.dump(count_vectors_labelled_train, os.path.join(dataset_path, dataset_name, count_labelled_train_name))
			joblib.dump(count_vectors_labelled_test, os.path.join(dataset_path, dataset_name, count_labelled_test_name))
			joblib.dump(count_vectors_unlabelled, os.path.join(dataset_path, dataset_name, count_unlabelled_test_name))

			joblib.dump(train_labels, os.path.join(dataset_path, dataset_name, 'train_labels'))
			joblib.dump(test_labels, os.path.join(dataset_path, dataset_name, 'test_labels'))
			joblib.dump(label_map, os.path.join(dataset_path, dataset_name, 'label_map'))

			joblib.dump(tfidf_vectorizer, os.path.join(dataset_path, dataset_name, tfidf_vectorizer_name))
			joblib.dump(count_vectorizer, os.path.join(dataset_path, dataset_name, count_vectorizer_name))

			joblib.dump(vocab, os.path.join(dataset_path, dataset_name, vocab_name))

			vectorized_labelled_train = tfidf_vectors_labelled_train if use_tfidf else count_vectors_labelled_train
			vectorized_labelled_test = tfidf_vectors_labelled_test if use_tfidf else count_vectors_labelled_test
			vectorized_unlabelled = tfidf_vectors_unlabelled if use_tfidf else count_vectors_unlabelled

	return (vectorized_labelled_train, train_labels, vectorized_labelled_test, test_labels, vectorized_unlabelled,
			label_map, labelled_features, vocab, count_vectorizer, tfidf_vectorizer)


def fetch_method51_classif_dataset_vectorized(dataset_path, dataset_name, use_tfidf=False, extraction_style='all',
											  binarize=False, tf_normalisation=False, ngram_range=(1, 2),
											  force_recreate_dataset=False):
	vectorized_labelled_train = None
	train_labels = None
	vectorized_labelled_test = None
	test_labels = None
	vectorized_unlabelled = None
	label_map = None
	labelled_features = None
	labelled_features_idx = None

	tfidf_labelled_train_name = 'tfidf_vectors_labelled_train_' + extraction_style if extraction_style != None else 'tfidf_vectors_labelled_train'
	tfidf_labelled_test_name = 'tfidf_vectors_labelled_test_' + extraction_style if extraction_style != None else 'tfidf_vectors_labelled_test'
	tfidf_unlabelled_name = 'tfidf_vectors_unlabelled_' + extraction_style if extraction_style != None else 'tfidf_vectors_unlabelled'
	count_labelled_train_name = 'count_vectors_labelled_train_' + extraction_style if extraction_style != None else 'count_vectors_labelled_train'
	count_labelled_test_name = 'count_vectors_labelled_test_' + extraction_style if extraction_style != None else 'count_vectors_labelled_test'
	count_unlabelled_test_name = 'count_vectors_unlabelled_' + extraction_style if extraction_style != None else 'count_vectors_unlabelled'

	if (binarize):
		tfidf_labelled_train_name += '_binary'
		tfidf_labelled_test_name += '_binary'
		tfidf_unlabelled_name += '_binary'
		count_labelled_train_name += '_binary'
		count_labelled_test_name += '_binary'
		count_unlabelled_test_name += '_binary'

	if (tf_normalisation):
		count_labelled_train_name += '_tf_norm'
		count_labelled_test_name += '_tf_norm'
		count_unlabelled_test_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		tfidf_labelled_train_name = '_'.join([tfidf_labelled_train_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_labelled_test_name = '_'.join([tfidf_labelled_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_unlabelled_name = '_'.join([tfidf_unlabelled_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_labelled_train_name = '_'.join([count_labelled_train_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_labelled_test_name = '_'.join([count_labelled_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_unlabelled_test_name = '_'.join([count_unlabelled_test_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	# Check if cached stuff exists
	if (not force_recreate_dataset and
			os.path.exists(os.path.join(dataset_path, dataset_name, tfidf_labelled_train_name if use_tfidf else count_labelled_train_name))):
		vectorized_labelled_train = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_labelled_train_name if use_tfidf else count_labelled_train_name))
		train_labels = joblib.load(os.path.join(dataset_path, dataset_name, 'train_labels'))
		vectorized_labelled_test = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_labelled_test_name if use_tfidf else count_labelled_test_name))
		test_labels = joblib.load(os.path.join(dataset_path, dataset_name, 'test_labels'))
		vectorized_unlabelled = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_unlabelled_name if use_tfidf else count_unlabelled_test_name))
		label_map = joblib.load(os.path.join(dataset_path, dataset_name, 'label_map'))
		labelled_features = joblib.load(os.path.join(dataset_path, dataset_name, 'labelled_features'))
		labelled_features_idx = joblib.load(os.path.join(dataset_path, dataset_name, 'labelled_features_idx'))
	else:
		if (os.path.exists(os.path.join(dataset_path, dataset_name))):
			labelled_train = json.load(open(os.path.join(dataset_path, dataset_name, 'training.json')))
			labelled_test = json.load(open(os.path.join(dataset_path, dataset_name, '-'.join([dataset_name, 'gold-standard.json']))))
			unlabelled = json.load(open(os.path.join(dataset_path, dataset_name, '-'.join([dataset_name, 'unlabelled.json']))))
			model = json.load(open(os.path.join(dataset_path, dataset_name, 'nbmodel.json')))

			label_map = dict((value, key) for key, value in dict(enumerate(model['labels'])).items())
			train_targets = [label_map[x['label']] for x in labelled_train]
			test_targets = [label_map[x['label']] for x in labelled_test]
			train_data = [x['text'] for x in labelled_train]
			test_data = [x['text'] for x in labelled_test]
			unlabelled_data = [x['text'] for x in unlabelled]

			# In the method51 framework, bigrams are represented w1_w2, so we need to replace the underscore with a space again (thats messy but good enough for now...)
			vocab = [v if not '_' in v else string.replace(v, '_', ' ') for v in model['vocab']]

			tfidf_vectorizer = TfidfVectorizer(decode_error='replace', vocabulary=vocab, ngram_range=ngram_range)
			count_vectorizer = CountVectorizer(decode_error='replace', vocabulary=vocab, binary=binarize, ngram_range=ngram_range)
			transformer = TfidfTransformer(use_idf=False)

			if (extraction_style == 'all'):
				all_data = labelled_train + unlabelled_data
				tfidf_vectorizer.fit(all_data)
				count_vectorizer.fit(all_data)
			else:
				tfidf_vectorizer.fit(labelled_train)
				count_vectorizer.fit(labelled_train)

			tfidf_vectors_labelled_train = tfidf_vectorizer.transform(train_data)
			tfidf_vectors_labelled_test = tfidf_vectorizer.transform(test_data)
			tfidf_vectors_unlabelled = tfidf_vectorizer.transform(unlabelled_data)

			if (tf_normalisation):
				count_vectors_labelled_train = transformer.fit_transform(count_vectorizer.transform(train_data))
				count_vectors_labelled_test = transformer.fit_transform(count_vectorizer.transform(test_data))
				count_vectors_unlabelled = transformer.fit_transform(count_vectorizer.transform(unlabelled_data))
			else:
				count_vectors_labelled_train = count_vectorizer.transform(train_data)
				count_vectors_labelled_test = count_vectorizer.transform(test_data)
				count_vectors_unlabelled = count_vectorizer.transform(unlabelled_data)

			labelled_features = {}
			labelled_features_idx = {}
			for l in model['labels']:
				if (l in list(model['labelFeatureAlphas'].keys())):
					labelled_features[l] = list(model['labelFeatureAlphas'][l].keys())
					idx_list = []
					for feat in list(model['labelFeatureAlphas'][l].keys()):
						clean_feat = string.replace(feat, '_', ' ')
						if (clean_feat in count_vectorizer.get_feature_names()):
							idx_list.append(count_vectorizer.get_feature_names().index(clean_feat))
					labelled_features_idx[l] = idx_list

			train_labels = np.array(train_targets)
			test_labels = np.array(test_targets)

			# Cache the stuff
			joblib.dump(tfidf_vectors_labelled_train, os.path.join(dataset_path, dataset_name, tfidf_labelled_train_name))
			joblib.dump(tfidf_vectors_labelled_test, os.path.join(dataset_path, dataset_name, tfidf_labelled_test_name))
			joblib.dump(tfidf_vectors_unlabelled, os.path.join(dataset_path, dataset_name, tfidf_unlabelled_name))

			joblib.dump(count_vectors_labelled_train, os.path.join(dataset_path, dataset_name, count_labelled_train_name))
			joblib.dump(count_vectors_labelled_test, os.path.join(dataset_path, dataset_name, count_labelled_test_name))
			joblib.dump(count_vectors_unlabelled, os.path.join(dataset_path, dataset_name, count_unlabelled_test_name))

			joblib.dump(train_labels, os.path.join(dataset_path, dataset_name, 'train_labels'))
			joblib.dump(test_labels, os.path.join(dataset_path, dataset_name, 'test_labels'))
			joblib.dump(label_map, os.path.join(dataset_path, dataset_name, 'label_map'))
			joblib.dump(labelled_features, os.path.join(dataset_path, dataset_name, 'labelled_features'))
			joblib.dump(labelled_features_idx, os.path.join(dataset_path, dataset_name, 'labelled_features_idx'))
			pickle.dump(train_data, open(os.path.join(dataset_path, dataset_name, 'train_data'), 'w'))
			pickle.dump(test_data, open(os.path.join(dataset_path, dataset_name, 'test_data'), 'w'))
			pickle.dump(unlabelled_data, open(os.path.join(dataset_path, dataset_name, 'unlabelled_data'), 'w'))

			vectorized_labelled_train = tfidf_vectors_labelled_train if use_tfidf else count_vectors_labelled_train
			vectorized_labelled_test = tfidf_vectors_labelled_test if use_tfidf else count_vectors_labelled_test
			vectorized_unlabelled = tfidf_vectors_unlabelled if use_tfidf else count_vectors_unlabelled

	return (vectorized_labelled_train, train_labels, vectorized_labelled_test, test_labels, vectorized_unlabelled,
			label_map, labelled_features, labelled_features_idx)


def fetch_twitter_fyp_dataset_vectorized(dataset_path, dataset_name, use_tfidf=False, wrap_in_list=False,
										 return_raw=False, extraction_style='all', binarize=False, tf_normalisation=False,
										 ngram_range=(1, 1), force_recreate_dataset=False):
	vectorized_labelled = None
	labels = None
	vectorized_unlabelled = None
	raw_labelled = None
	raw_unlabelled = None

	tfidf_vectorizer = TfidfVectorizer(decode_error='replace', ngram_range=ngram_range)
	count_vectorizer = CountVectorizer(decode_error='replace', binary=binarize, ngram_range=ngram_range)
	transformer = TfidfTransformer(use_idf=False)

	tfidf_labelled_name = 'tfidf_vectors_labelled_' + extraction_style if extraction_style != None else 'tfidf_vectors_labelled'
	tfidf_unlabelled_name = 'tfidf_vectors_unlabelled_' + extraction_style if extraction_style != None else 'tfidf_vectors_unlabelled'
	count_labelled_name = 'count_vectors_labelled_' + extraction_style if extraction_style != None else 'count_vectors_labelled'
	count_unlabelled_name = 'count_vectors_unlabelled_' + extraction_style if extraction_style != None else 'count_vectors_unlabelled'

	if (binarize):
		tfidf_labelled_name += '_binary'
		tfidf_unlabelled_name += '_binary'
		count_labelled_name += '_binary'
		count_unlabelled_name += '_binary'

	if (tf_normalisation):
		count_labelled_name += '_tf_norm'
		count_unlabelled_name += '_tf_norm'

	if (not min(ngram_range) == max(ngram_range) == 1):
		tfidf_labelled_name = '_'.join([tfidf_labelled_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		tfidf_unlabelled_name = '_'.join([tfidf_unlabelled_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_labelled_name = '_'.join([count_labelled_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])
		count_unlabelled_name = '_'.join([count_unlabelled_name, 'ngram_range', str(ngram_range[0]), str(ngram_range[1])])

	# Check if cached stuff exists
	if (not force_recreate_dataset and
			os.path.exists(os.path.join(dataset_path, dataset_name, tfidf_labelled_name if use_tfidf else
			count_labelled_name)) and os.path.exists(os.path.join(dataset_path, dataset_name, 'raw_labelled_data'))):

		vectorized_labelled = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_labelled_name if use_tfidf else
			count_labelled_name))

		labels = joblib.load(os.path.join(dataset_path, dataset_name, 'labels'))

		vectorized_unlabelled = joblib.load(os.path.join(dataset_path, dataset_name, tfidf_unlabelled_name
			if use_tfidf else count_unlabelled_name))

		raw_labelled = joblib.load(os.path.join(dataset_path, dataset_name, 'raw_labelled_data'))
		raw_unlabelled = joblib.load(os.path.join(dataset_path, dataset_name, 'raw_unlabelled_data'))
	else:
		if (os.path.exists(os.path.join(dataset_path, dataset_name))):
			data_file = '%s.pkz' % (dataset_name,)
			#label_map_file = '%s_map' % (dataset_name,)
			with open(os.path.join(dataset_path, dataset_name, data_file), 'rB') as dataset:
				labelled_data = pickle.load(dataset)[dataset_name]
				labels = pickle.load(dataset)[dataset_name]
				unlabelled_data = pickle.load(dataset)[dataset_name]
				#label_map = pickle.loads(open(os.path.join(dataset_path, dataset_name, label_map_file), 'rb').read().decode('zip'))

				if (extraction_style == 'all'):
					all_data = labelled_data + unlabelled_data
					tfidf_vectorizer.fit(all_data)
					count_vectorizer.fit(all_data)
				else:
					tfidf_vectorizer.fit(labelled_data)
					count_vectorizer.fit(labelled_data)

				tfidf_vectors_labelled = tfidf_vectorizer.fit_transform(labelled_data)
				tfidf_vectors_unlabelled = tfidf_vectorizer.transform(unlabelled_data)

				if (tf_normalisation):
					count_vectors_labelled = transformer.fit_transform(count_vectorizer.transform(labelled_data))
					count_vectors_unlabelled = transformer.fit_transform(count_vectorizer.transform(unlabelled_data))
				else:
					count_vectors_labelled = count_vectorizer.transform(labelled_data)
					count_vectors_unlabelled = count_vectorizer.transform(unlabelled_data)

				# Cache vectorized
				joblib.dump(tfidf_vectors_labelled, os.path.join(dataset_path, dataset_name, tfidf_labelled_name))
				joblib.dump(count_vectors_labelled, os.path.join(dataset_path, dataset_name, count_labelled_name))
				joblib.dump(tfidf_vectors_unlabelled, os.path.join(dataset_path, dataset_name, tfidf_unlabelled_name))
				joblib.dump(count_vectors_unlabelled, os.path.join(dataset_path, dataset_name, count_unlabelled_name))
				joblib.dump(labels, os.path.join(dataset_path, dataset_name, 'labels'))
				joblib.dump(tfidf_vectorizer, os.path.join(dataset_path, dataset_name, 'tfidf_vectorizer'))
				joblib.dump(count_vectorizer, os.path.join(dataset_path, dataset_name, 'count_vectorizer'))
				#pickle.dump(labelled_data, open(os.path.join(dataset_path, dataset_name, 'labelled_data'), 'w'))
				#pickle.dump(unlabelled_data, open(os.path.join(dataset_path, dataset_name, 'unlabelled_data'), 'w'))

				# Cache raw
				joblib.dump(labelled_data, os.path.join(dataset_path, dataset_name, 'raw_labelled_data'))
				joblib.dump(unlabelled_data, os.path.join(dataset_path, dataset_name, 'raw_unlabelled_data'))

				vectorized_labelled = tfidf_vectors_labelled if use_tfidf else count_vectors_labelled
				vectorized_unlabelled = tfidf_vectors_unlabelled if use_tfidf else count_vectors_unlabelled

	if (return_raw):
		return (raw_labelled, labels, raw_unlabelled)

	return (vectorized_labelled, labels, vectorized_unlabelled) if not wrap_in_list else [(vectorized_labelled, labels, vectorized_unlabelled)]


def fetch_google_news_word2vec_300dim_vectors(dataset_path=os.path.join(paths.get_dataset_path(), 'google_news')):
	return Word2Vec.load_word2vec_format(os.path.join(dataset_path, 'GoogleNews-vectors-negative300.bin'), binary=True)


def fetch_stanford_sentiment_treebank_dataset(dataset_path=os.path.join(paths.get_dataset_path(), 'stanford_sentiment_treebank'),
											  fine_grained=False, vectorisation='count', force_recreate_dataset=False,
											  vectorisation_opts={'ngram_range': (1, 2), 'extraction_style': 'all', 'binarize': False,
																  'tf_normalisation': False}):
	'''
	:param dataset_path: Path to dataset on disk
	:param fine_grained: Whether polarity (2-classes) or fine-grained (5-classes) Sentiment dataset is used
	:param vectorisation: 'count' for standard BoW vectors, 'tfidf' for TF-IDF vectors, 'word2vec' for word2vec embeddings and 'glove' for GloVe embeddings; default is 'count'
	:param vectorisation_opts: Options for the vectorisation, e.g. ngram_range for 'count' and 'tfidf', various word2vec params such as 'sg' or 'hs', number of negative samples, etc
	:param force_recreate_dataset: Whether or not the dataset should be recreated
	:return: A Dataset
	'''

	if (path_utils.check_all_exist(dataset_path, ['train_data', 'valid_data', 'test_data', 'y_train', 'y_valid', 'y_test'])):
		y_train = joblib.load(os.path.join(dataset_path, 'y_train'))
		y_valid = joblib.load(os.path.join(dataset_path, 'y_valid'))
		y_test = joblib.load(os.path.join(dataset_path, 'y_test'))
		train_data = joblib.load(os.path.join(dataset_path, 'train_data'))
		valid_data = joblib.load(os.path.join(dataset_path, 'valid_data'))
		test_data = joblib.load(os.path.join(dataset_path, 'test_data'))
	else:
		# Create sub_path out of options
		fine_grained_subpath = '2_class' if not fine_grained else '5_class'
		path = os.path.join(dataset_path, fine_grained_subpath)
		for key, value in vectorisation_opts.iteritems():
			clean_value = str(value).replace(' ', '').replace(',', '-').replace('(', '').replace(')', '').replace('[', '').replace(']', '').strip()
			path = os.path.join(path, '_'.join([key, clean_value]))

		label_dict = collections.defaultdict(list)
		sent_dict = collections.defaultdict(list)

		phrases = read_su_sentiment_rotten_tomatoes(dirname=os.path.join(paths.get_dataset_path(), 'stanford_sentiment_treebank'))

		for phrase in phrases:
			if (phrase.split is not None and phrase.sentence_id is not None):
				label_dict[phrase.split].append(_stanford_stb_fine_grained_label_mapping(phrase.sentiment))
				sent_dict[phrase.split].append(phrase.words)

		y_train = np.array(label_dict['train'])
		y_valid = np.array(label_dict['dev'])
		y_test = np.array(label_dict['test'])
		train_data = sent_dict['train']
		valid_data = sent_dict['dev']
		test_data = sent_dict['test']

		joblib.dump(y_train, os.path.join(dataset_path, 'y_train'))
		joblib.dump(y_valid, os.path.join(dataset_path, 'y_valid'))
		joblib.dump(y_test, os.path.join(dataset_path, 'y_test'))
		joblib.dump(train_data, os.path.join(dataset_path, 'train_data'))
		joblib.dump(valid_data, os.path.join(dataset_path, 'valid_data'))
		joblib.dump(test_data, os.path.join(dataset_path, 'test_data'))

	return (train_data, y_train, valid_data, y_valid, test_data, y_test)


def _stanford_stb_fine_grained_label_mapping(sentiment_score):
	if (sentiment_score <= 0.2):
		return 0
	elif (sentiment_score > 0.2 and sentiment_score <= 0.4):
		return 1
	elif (sentiment_score > 0.4 and sentiment_score <= 0.6):
		return 2
	elif (sentiment_score > 0.6 and sentiment_score <= 0.8):
		return 3
	elif (sentiment_score > 0.8):
		return 4


def fetch_huang_et_al_2012_vectors(dataset_path=os.path.join(paths.get_dataset_path(), 'huang_et_al_2012'), original=False):
	vectors_path = os.path.join(dataset_path, 'embeddings', 'wordreps_orig.mat' if original else 'wordreps.mat')
	vocab_path = os.path.join(dataset_path, 'embeddings', 'vocab.mat')

	# TODO: Stuff
	V = loadmat(vocab_path)

	W = loadmat(vectors_path)

	return None


def fetch_collobert_and_weston_vectors(dataset_path=os.path.join(paths.get_dataset_path(), 'collobert_weston_original'), target_words=None):
	index = None
	W = None

	if (target_words is None): # Load all
		# Build inverted word index
		with open(os.path.join(dataset_path, 'senna', 'hash', 'words.lst')) as f_word_list:
			word_list = list(map(lambda w: w.strip(), f_word_list.read().split('\n')))

		index = dict(zip(word_list, range(len(word_list))))

		W = np.loadtxt(os.path.join(dataset_path, 'senna', 'embeddings', 'embeddings.txt'))
	else:
		with open(os.path.join(dataset_path, 'senna', 'hash', 'words.lst')) as f_word_list:
			word_list = list(map(lambda w: w.strip(), f_word_list.read().split('\n')))

		all_index = dict(zip(word_list, range(len(word_list))))
		index = dict(zip(target_words), range(len(target_words)))

		rng = []
		for w in target_words:
			if (w in all_index):
				rng.append(all_index[w])
			else:
				rng.append(all_index['UNKNOWN'])

		W = np.loadtxt(os.path.join(dataset_path, 'senna', 'embeddings', 'embeddings.txt'))
		W = W[np.array(rng)]

	return (index, W)


def fetch_scws_wikipedia_apt_vectors(example_id, dataset_path=os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'cached_vectors'),
									 dep_order=2, normalised=True, exclude_contexts=False, use_lemma=False, use_pos=False, use_pmi=False,
									 cache_if_not_exists=True, composition_order=1, coarse=False, check_ext_cache=False, logging=None):
	fname_1 = '1.cached_ctx_vecs-{}{}.json.gz'.format(dep_order, '-norm' if normalised else '')
	fname_2 = '2.cached_ctx_vecs-{}{}.json.gz'.format(dep_order, '-norm' if normalised else '')
	subpath = 'wiki_lc_{}{}{}'.format(dep_order, '_norm' if normalised else '', '_coarse' if coarse else '')

	# Check for global cache
	ext_cache_name = 'wikipedia_lc_{}{}_lemma-{}_pos-{}{}_vectors_cache.joblib'.format(dep_order, '_norm' if normalised else '', use_lemma, use_pos, '_pmi' if use_pmi else '')
	ext_cache_path = os.path.join(paths.get_external_dataset_path(), 'word_similarity_in_ctx', 'cached_vectors', ext_cache_name)
	if (check_ext_cache and ext_cache_path):
		if (logging is not None):
			logging.info('[{}] - Loading cache from file: {}...'.format(strftime('%H:%M:%S', gmtime()), ext_cache_name))
		return joblib.load(ext_cache_path)

	if (os.path.exists(os.path.join(dataset_path, subpath, str(example_id), fname_1)) and os.path.exists(os.path.join(dataset_path, subpath, str(example_id), fname_2))):
		with gzip.open(os.path.join(dataset_path, subpath, str(example_id), fname_1), 'rt') as vec_cache:
			vectors_1 = json.loads(vec_cache.read())

		with gzip.open(os.path.join(dataset_path, subpath, str(example_id), fname_2), 'rt') as vec_cache:
			vectors_2 = json.loads(vec_cache.read())
	else:
		target_ctx_1 = []
		target_ctx_2 = []
		target_ctx_path = os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'extracted_contexts{}'.format('-lemma' if use_lemma else '')
									   , str(example_id))
		with open(os.path.join(target_ctx_path, '1.txt'), 'r') as f_ctx_1, open(os.path.join(target_ctx_path, '2.txt'), 'r') as f_ctx_2:
			extracted_ctx_1 = f_ctx_1.read().strip()
			extracted_ctx_2 = f_ctx_2.read().strip()
			ctx_1 = extracted_ctx_1.split('\t')[1]
			ctx_2 = extracted_ctx_2.split('\t')[1]
			target_word_1 = extracted_ctx_1.strip().split('\t')[0].lower()
			target_word_2 = extracted_ctx_2.split('\t')[0].lower()

			if (not exclude_contexts):
				buffer = ''
				for ctx, target_ctx in zip([ctx_1, ctx_2], [target_ctx_1, target_ctx_2]):
					for c in ctx.split(','):
						if ('_' not in c):
							buffer += c
						else:
							word, rel = c.rsplit('_', 1)

							if (buffer != ''):
								word = '{},{}'.format(buffer, word)
								buffer = ''

							if (len(rel.split('.')) <= composition_order):
								target_ctx.append((word.lower(), rel))

		# Extract target & context vectors
		target_words_1 = [t[0] for t in target_ctx_1]
		target_words_2 = [t[0] for t in target_ctx_2]

		if (target_word_1 not in target_words_1):
			target_words_1.append(target_word_1)

		if (target_word_2 not in target_words_2):
			target_words_2.append(target_word_2)

		vec_path = os.path.join(paths.get_dataset_path(), 'wikipedia', 'vectors')
		vector_in_file = 'wikipedia_lc_{}{}{}_lemma-{}_pos-{}{}_vectors.tsv.gz'.format(dep_order, '_coarse' if coarse else '', '_norm' if normalised else '', use_lemma, use_pos, '_pmi' if use_pmi else '')

		print('Loading Vectors from File={}; Full path={}'.format(vector_in_file, os.path.join(vec_path, vector_in_file)))

		# Cache CTXs, load target words commonly to reduce i/o load a bit and split them later
		vectors = vector_utils.load_csv_vectors(os.path.join(vec_path, vector_in_file), words=list(set(target_words_1) | set(target_words_2)), out_prefix='\t', mod_logging_freq=3000)

		vectors_1 = {}
		for w in target_words_1:
			if (w in vectors):
				vectors_1[w] = vectors[w]

		vectors_2 = {}
		for w in target_words_2:
			if (w in vectors):
				vectors_2[w] = vectors[w]

		# Caching requires a shitton of space, hence its optional
		if (cache_if_not_exists):
			out_path = os.path.join(dataset_path, subpath, str(example_id))

			if (not os.path.exists(out_path)):
				os.makedirs(out_path)

			print('Caching vector file: {}'.format(os.path.join(out_path, fname_1)))
			with gzip.open(os.path.join(out_path, fname_1), 'wt') as vec_dump:
				vec_dump.write(json.dumps(vectors_1))

			print('Caching vector file: {}'.format(os.path.join(out_path, fname_2)))
			with gzip.open(os.path.join(out_path, fname_2), 'wt') as vec_dump:
				vec_dump.write(json.dumps(vectors_2))

	return (vectors_1, vectors_2)


def fetch_scws_dataset(dataset_path=os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx'), dataset_option='raw'):

	if (dataset_option == 'raw'):
		sents = []
		with open(os.path.join(dataset_path, 'ratings.txt'), mode='r') as raw_dataset:
			for line in raw_dataset:
				sents.append(line.strip())

		return sents
	elif (dataset_option == 'apt_preprocessed'):
		sents = []
		with open(os.path.join(dataset_path, 'apt_contexts_extracted', 'ratings.tsv'), mode='r') as apt_preprocessed:
			for line in apt_preprocessed:
				sents.append(line.strip())

		return sents
	else:
		#if (os.path.exists(datset_path)):
		#	pass
		#else:
		if (not os.path.exists(os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'processed'))):
			os.makedirs(os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'processed'))
		with open(os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'ratings.txt'), 'rb') as raw_dataset, \
			open(os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'processed', 'words.csv'), 'wb') as csv_words, \
			open(os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'processed', 'words_incl_pos.csv'), 'wb') as csv_words_incl_pos:

			csv_words_writer = csv.writer(csv_words)
			csv_words_incl_pos_writer = csv.writer(csv_words_incl_pos)

			# Preprocess
			for line in raw_dataset:
				parts = line.split('\t')
				row_id = parts[0]
				word1 = parts[1]
				word1_incl_pos = '%s\\%s' % (word1, parts[2])
				word2 = parts[3]
				word2_incl_pos = '%s\\%s'  % (word2, parts[4])
				ctx1 = parts[5].replace('<b>', '').replace('</b>', '').replace('</ b>', '')
				ctx2 = parts[6].replace('<b>', '').replace('</b>', '').replace('</ b>', '')
				avg_sim_rating = float(parts[7])

				words_incl_pos = [row_id, word1_incl_pos, word2_incl_pos, avg_sim_rating]
				words = [row_id, word1, word2, avg_sim_rating]

				# Write Words CSV
				csv_words_writer.writerow(words)
				csv_words_incl_pos_writer.writerow(words_incl_pos)

				# Write Sentences
				os.makedirs(os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'processed', row_id))
				for ctx, ctx_name in zip([ctx1, ctx2], ['ctx1.txt', 'ctx2.txt']):
					f = open(os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'processed', row_id, ctx_name), 'wb')
					f.write(ctx)
					f.close()

			# Dependency Parse with Stanford Parser
			'''
			for folder in os.listdir(os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'processed')):
				processed_path = os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'processed', folder)
				parsed_path = os.path.join(paths.get_dataset_path(), 'word_similarity_in_ctx', 'parsed', folder)
				if (os.path.isdir(processed_path)):
					if (not os.path.exists(parsed_path)):
						os.makedirs(parsed_path)
					stanford_utils.run_stanford_pipeline(processed_path, os.path.join(paths.get_base_path(), 'stanford-corenlp'), java_threads=4, filelistdir="")
			'''
