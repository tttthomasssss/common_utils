__author__ = 'thk22'

import collections
import glob
import json
import os
import re
import sgmllib
import sys


class AptemodIndex(object):
	def __init__(self, aptemod_path):
		self._aptemod_base_path = aptemod_path
		self._doc_index = collections.defaultdict(lambda: collections.defaultdict(list))
		self._topic_index = collections.defaultdict(list)
		self._topic_distribution = collections.defaultdict(int)
		self._country_index = collections.defaultdict(list)
		self._country_distribution = collections.defaultdict(int)
		self._subset_index = collections.defaultdict(list)
		self._subset_distribution = collections.defaultdict(int)

	def add_doc(self, doc_id, topic_list, country_list, doc_path, subset):
		self._doc_index[doc_id]['topics'].extend(topic_list)
		self._doc_index[doc_id]['countries'].extend(country_list)
		self._doc_index[doc_id]['path'].append(doc_path)
		self._doc_index[doc_id]['subset'].append(subset)

		for topic in topic_list:
			self._topic_index[topic].append(doc_id)
			self._topic_distribution[topic] += 1

		for country in country_list:
			self._country_index[country].append(doc_id)
			self._country_distribution[country] += 1

		self._subset_index[subset].append(doc_id)
		self._subset_distribution[subset] += 1

	def store_index(self):
		print('Storing Doc Index...')
		doc_idx_file = open(os.path.join(self._aptemod_base_path, 'doc_idx.json'), 'w')
		json.dump(self._doc_index, doc_idx_file)
		doc_idx_file.close()

		print('Storing Topic Index...')
		topic_idx_file = open(os.path.join(self._aptemod_base_path, 'topic_idx.json'), 'w')
		json.dump(self._topic_index, topic_idx_file)
		topic_idx_file.close()

		print('Storing Topic Distribution...')
		topic_distribution_file = open(os.path.join(self._aptemod_base_path, 'topic_distribution.json'), 'w')
		json.dump(self._topic_distribution, topic_distribution_file)
		topic_distribution_file.close()

		print('Storing Country Index...')
		country_idx_file = open(os.path.join(self._aptemod_base_path, 'country_idx.json'), 'w')
		json.dump(self._country_index, country_idx_file)
		country_idx_file.close()

		print('Storing Country Distribution...')
		country_distro_file = open(os.path.join(self._aptemod_base_path, 'country_distribution.json'), 'w')
		json.dump(self._country_distribution, country_distro_file)
		country_distro_file.close()

		print('Storing Subset Index...')
		subset_idx_file = open(os.path.join(self._aptemod_base_path, 'subset_idx.json'), 'w')
		json.dump(self._subset_index, subset_idx_file)
		subset_idx_file.close()

		print('Storing Subset Distribution...')
		subset_distro_file = open(os.path.join(self._aptemod_base_path, 'subset_distribution.json'), 'w')
		json.dump(self._subset_distribution, subset_distro_file)
		subset_distro_file.close()

	def load_index(self):
		doc_idx_file = open(os.path.join(self._aptemod_base_path, 'doc_idx.json'), 'r')
		self._doc_index = json.load(doc_idx_file)
		doc_idx_file.close()

		topic_idx_file = open(os.path.join(self._aptemod_base_path, 'topic_idx.json'), 'r')
		self._topic_index = json.load(topic_idx_file)
		topic_idx_file.close()

		topic_distribution_file = open(os.path.join(self._aptemod_base_path, 'topic_distribution.json'), 'r')
		self._topic_distribution = json.load(topic_distribution_file)
		topic_distribution_file.close()

		subset_idx_file = open(os.path.join(self._aptemod_base_path, 'subset_idx.json'), 'r')
		self._subset_index = json.load(subset_idx_file)
		subset_idx_file.close()

		subset_distribution_file = open(os.path.join(self._aptemod_base_path, 'subset_distribution.json'), 'r')
		self._subset_distribution = json.load(subset_distribution_file)
		subset_distribution_file.close()

		country_idx_file = open(os.path.join(self._aptemod_base_path, 'country_idx.json'), 'r')
		self._country_index = json.load(country_idx_file)
		country_idx_file.close()

		country_distribution_file = open(os.path.join(self._aptemod_base_path, 'country_distribution.json'), 'r')
		self._country_distribution = json.load(country_distribution_file)
		country_distribution_file.close()

	def topic_list_for_doc_id(self, doc_id):
		return self._doc_index[doc_id]['topics']

	def country_list_for_doc_id(self, doc_id):
		return self._doc_index[doc_id]['countries']

class AptemodParser(sgmllib.SGMLParser):
	'''Utility class to parse a SGML file and yield documents one at a time.'''
	def __init__(self, verbose=0):
		sgmllib.SGMLParser.__init__(self, verbose)
		self._reset()

	def _reset(self):
		self.in_title = False
		self.in_body = False
		self.in_topics = False
		self.in_topic_d = False
		self.in_places = False
		self.in_places_d = False
		self.title = ''
		self.body = ''
		self.topics = []
		self.places = []
		self.topic_d = ''
		self.place_d = ''
		self.aptemod = False
		self.subset = ''
		self.doc_id = ''

	def parse(self, fd):
		self.docs = []
		for chunk in fd:
			self.feed(chunk)
			for doc in self.docs:
				yield doc
			self.docs = []
		self.close()

	def handle_data(self, data):
		if (self.in_body):
			self.body += data
		elif (self.in_title):
			self.title += data
		elif (self.in_topic_d):
			self.topic_d += data
		elif (self.in_places_d):
			self.place_d += data

	def start_reuters(self, attributes):
		attr_dict = dict(attributes)

		if ('lewissplit' in attr_dict):
			self.aptemod = True
			self.subset = attr_dict['lewissplit']
			self.doc_id = attr_dict['newid']

	def end_reuters(self):
		if (self.aptemod):
			self.body = re.sub(r'\s+', r' ', self.body)
			self.docs.append({'title': self.title,
							  'body': self.body,
							  'topics': self.topics,
							  'subset': self.subset,
							  'places': self.places,
							  'doc_id': self.doc_id})

		self._reset()

	def start_title(self, attributes):
		self.in_title = True

	def end_title(self):
		self.in_title = False

	def start_body(self, attributes):
		self.in_body = True

	def end_body(self):
		self.in_body = False

	def start_topics(self, attributes):
		self.in_topics = True

	def end_topics(self):
		self.in_topics = False

	def start_places(self, attributes):
		self.in_places = True

	def end_places(self):
		self.in_places = False

	def start_d(self, attributes):
		if (self.in_topics):
			self.in_topic_d = True
		elif (self.in_places):
			self.in_places_d = True

	def end_d(self):
		if (self.in_topics):
			self.in_topic_d = False
			self.topics.append(self.topic_d)
			self.topic_d = ''
		elif (self.in_places):
			self.in_places_d = False
			self.places.append(self.place_d)
			self.place_d = ''

def create_index(dataset_path):
	parser = AptemodParser()
	index = AptemodIndex(dataset_path)

	train_data = open(os.path.join(dataset_path, 'aptemod_train.tsv'), 'w')
	test_data = open(os.path.join(dataset_path, 'aptemod_test.tsv'), 'w')
	unlabelled_data = open(os.path.join(dataset_path, 'aptemod_unlabelled.tsv'), 'w')

	for filename in glob.glob(os.path.join(dataset_path, '*.sgm')):
		for doc in parser.parse(open(filename)):
			index.add_doc(doc['doc_id'], doc['topics'], doc['places'], filename, doc['subset'])
			line = '%s\t%s\n' % (doc['doc_id'], doc['body'])
			if (doc['subset'].lower() == 'train'):
				train_data.write(line)
			elif (doc['subset'].lower() == 'test'):
				test_data.write(line)
			else:
				unlabelled_data.write(line)

	index.store_index()

	train_data.close()
	test_data.close()
	unlabelled_data.close()

if (__name__ == '__main__'):
	create_index(sys.argv[1] if len(sys.argv) > 1 else '/Volumes/LocalDataHD/thk22/DevSandbox/InfiniteSandbox/_datasets/aptemod/reuters21578')