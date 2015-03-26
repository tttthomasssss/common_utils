__author__ = 'thk22'
import collections
import json
import operator
import os

from sklearn.externals import joblib
import pymysql

from common import dataset_utils
from common import paths

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))


def load_raw_model_data(model_name):
	db_credentials = json.load(open(os.path.join(PROJECT_PATH, 'private_resources', 'mysql_credentials.json')))

	# Paths
	base_path = os.path.join(paths.get_dataset_path(), 'method51', 'models')

	sample_path = os.path.join(base_path, '%s-sample.json' % (model_name,))
	datasource_path = os.path.join(base_path, '%s-data-source.json' % (model_name,))
	labelling_path = os.path.join(base_path, '%s-labelling.json' % (model_name,))
	training_path = os.path.join(base_path, model_name, 'training.json')
	nb_model_path = os.path.join(base_path, model_name, 'nbmodel.json')

	# JSON data
	print '\tParsing the JSON model files...'
	sample = json.load(open(sample_path, 'rb'))
	datasource = json.load(open(datasource_path, 'rb'))
	labelling = json.load(open(labelling_path, 'rb'))
	nb_model = json.load(open(nb_model_path, 'rb'))
	try:
		training = json.load(open(training_path, 'rb'))
	except UnicodeDecodeError:
		training = json.load(open(training_path, 'rb'), encoding='latin-1')

	db_credentials['db'] = datasource['source_database']

	not_in_ids = []

	label_map = dict((value, key) for key, value in dict(enumerate(labelling['labels'])).iteritems())
	training_docs = []
	labels = []

	# Training Data
	print '\tGathering the labelled training data...'
	for t in training:
		training_docs.append(t['text'])
		not_in_ids.append(t['id'])
		labels.append(label_map[t['label']])

	# Labelled Features
	labelled_features = nb_model['labelFeatureAlphas']

	# Vocab
	vocab = nb_model['vocab']

	# DB data

	# Gold Standard Data
	print '\tGathering the Gold Standard data...'
	gs_docs = []
	gs_labels = []

	for k, v in labelling['sample_labels'].iteritems():
		item_label_distro = collections.defaultdict(int)
		for vv in v.itervalues():
			item_label_distro[vv] += 1

		max_label = max(item_label_distro.iteritems(), key=operator.itemgetter(1))[0]

		for s in sample:
			if (s['id'] == k):
				not_in_ids.append(k)
				gs_docs.append(s['text'])
				gs_labels.append(label_map[max_label])
				break

	# Unlabelled Data
	print '\tFetching the Unlabelled data...'
	unlabelled_docs = []
	not_in_ids_str = ','.join(not_in_ids)

	conn = pymysql.connect(**db_credentials)
	cur = conn.cursor()

	cur.execute("""SELECT * FROM `%s` WHERE id NOT IN (%s);""" % (datasource['document_source'], not_in_ids_str))

	for row in cur:
		unlabelled_docs.append(row[1])

	# Create Paths
	out_path = os.path.join(paths.get_dataset_path(), 'ws_paper', model_name)
	if (not os.path.exists(out_path)):
		os.makedirs(out_path)

	print '\tStarting the dump...'
	joblib.dump(training_docs, os.path.join(out_path, 'raw_training_docs'))
	joblib.dump(labels, os.path.join(out_path, 'raw_training_labels'))
	joblib.dump(label_map, os.path.join(out_path, 'label_map'))
	joblib.dump(labelled_features, os.path.join(out_path, 'labelled_features'))
	joblib.dump(vocab, os.path.join(out_path, 'vocab'))
	joblib.dump(gs_docs, os.path.join(out_path, 'raw_gold_standard_docs'))
	joblib.dump(gs_labels, os.path.join(out_path, 'raw_gold_standard_labels'))
	joblib.dump(unlabelled_docs, os.path.join(out_path, 'raw_unlabelled_docs'))
	print 'Finished!'


def convert_raw_model_data(model_name):
	print '\tFetching model...'
	data = dataset_utils.fetch_ws_paper_dataset_vectorized(os.path.join(paths.get_dataset_path(), 'ws_paper'), dataset_name=model_name, extraction_style='all')

	print '\tReturned %d items!' % (len(data),)
	print 'Finished!'

if (__name__ == '__main__'):
	#models = ['clacton-master-filtered', 'floodingbtr', 'alessia-cameron2', 'cameronboocheer']
	#models = ['isisenglishsubvertednot', 'nato1', 'amessagefromisustous-subversion']
	models = ['duggan-relevent-no-news', 'nato2subjects']

	for m in models:
		print 'Processing %s...' % (m,)
		load_raw_model_data(m)

		print 'Converting %s...'  % (m,)
		convert_raw_model_data(m)