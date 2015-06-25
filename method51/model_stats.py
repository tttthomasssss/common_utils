
import collections
import json
import operator
import os

from common import paths
import pymysql
from functools import reduce

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))


def _calculate_training_label_distribution(labels, training_data):
	label_distro = {}
	for label in labels: # Don't want to use a defaultdict as it might happen that the training data don't contain all of the labels
		label_distro[label] = 0.

	if (len(training_data) > 0):
		for x in training_data:
			label_distro[x['label']] += 1.

		sumsi = sum(label_distro.values())

		for l in label_distro.keys():
			label_distro[l] /= sumsi

	return label_distro


def _calculate_gold_standard_label_distribution(labels, sample_labels):
	label_distro = collections.defaultdict(float)

	if (len(sample_labels) > 0):
		for v in sample_labels.values():
			item_label_distro = collections.defaultdict(int)
			for vv in v.values():
				item_label_distro[vv] += 1

			label_distro[max(iter(item_label_distro.items()), key=operator.itemgetter(1))[0]] += 1.

		sumsi = sum(label_distro.values())

		for l in label_distro.keys():
			label_distro[l] /= sumsi

	return label_distro


def print_stats_for_models_at_path(p, min_unlabelled=0, min_labelled_training=0, min_gold_standard=0, min_num_labels=0):
	db_credentials = json.load(open(os.path.join(PROJECT_PATH, 'private_resources', 'mysql_credentials.json')))

	model_stats = {}

	for idx, subdir in enumerate([xx for xx in [x[0] for x in os.walk(p)] if xx != p]):
		_, model_name = os.path.split(subdir)

		model_stats[model_name] = {}

		sample_path = os.path.join(p, '%s-sample.json' % (model_name,))
		datasource_path = os.path.join(p, '%s-data-source.json' % (model_name,))
		labelling_path = os.path.join(p, '%s-labelling.json' % (model_name,))
		training_path = os.path.join(p, model_name, 'training.json')

		if (reduce(lambda acc, x: acc and os.path.exists(x), [sample_path, datasource_path, labelling_path, training_path], True)):
			sample = json.load(open(sample_path, 'rb'))
			datasource = json.load(open(datasource_path, 'rb'))
			labelling = json.load(open(labelling_path, 'rb'))
			try:
				training = json.load(open(training_path, 'rb'))
			except UnicodeDecodeError:
				training = json.load(open(training_path, 'rb'), encoding='latin-1')

			db_credentials['db'] = datasource['source_database']

			try:
				conn = pymysql.connect(**db_credentials)
				cur = conn.cursor()

				cur.execute("""SELECT COUNT(*) FROM `%s`;""" % (datasource['document_source'],))
				num_unlabelled = cur.fetchone()[0]

				num_labels = len(labelling['labels'])
				labels = labelling['labels']

				num_labelled_training = len(training)
				training_label_distribution = _calculate_training_label_distribution(labels, training)

				num_gold_standard = len(labelling['sample_labels'])
				gold_standard_label_distribution = _calculate_gold_standard_label_distribution(labels, labelling['sample_labels'])

				if (num_unlabelled > min_unlabelled and num_labelled_training > min_labelled_training and num_gold_standard > min_gold_standard and num_labels > min_num_labels):
					model_stats[model_name]['num_unlabelled'] = num_unlabelled
					model_stats[model_name]['num_labels'] = num_labels
					model_stats[model_name]['labels'] = labels
					model_stats[model_name]['num_labelled_training'] = num_labelled_training
					model_stats[model_name]['training_label_distribution'] = training_label_distribution
					model_stats[model_name]['num_gold_standard'] = num_gold_standard
					model_stats[model_name]['gold_standard_label_distribution'] = gold_standard_label_distribution

				#print model_stats[model_name]
			except pymysql.err.ProgrammingError as e:
				print(e)
			except pymysql.err.InternalError as e:
				print(e)
			finally:
				cur.close()
				if (conn.open):
					conn.close()

	print(json.dumps(model_stats))

if (__name__ == '__main__'):
	print_stats_for_models_at_path(os.path.join(paths.get_dataset_path(), 'method51', 'models'), min_num_labels=4)
	print_stats_for_models_at_path(os.path.join(paths.get_dataset_path(), 'method51', 'models-1'), min_num_labels=4)