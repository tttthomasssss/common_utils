__author__ = 'thk22'

import collections
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile


class RCV1Index(object):
	def __init__(self, rcv_path):
		self._rcv_base_path = rcv_path
		self._doc_index = collections.defaultdict(lambda: collections.defaultdict(list))
		self._topic_index = collections.defaultdict(list)
		self._country_index = collections.defaultdict(list)
		self._industry_index = collections.defaultdict(list)
		self._topic_distribution = {}
		self._industry_distribution = {}
		self._country_distribution = {}

	def load_index(self):
		doc_idx_file = open(os.path.join(self._rcv_base_path, 'doc_idx.json'), 'r')
		self._doc_index = json.load(doc_idx_file)
		doc_idx_file.close()

		topic_idx_file = open(os.path.join(self._rcv_base_path, 'topic_idx.json'), 'r')
		self._topic_index = json.load(topic_idx_file)
		topic_idx_file.close()

		topic_distribution_file = open(os.path.join(self._rcv_base_path, 'topic_distribution.json'), 'r')
		self._topic_distribution = json.load(topic_distribution_file)
		topic_distribution_file.close()

		industry_idx_file = open(os.path.join(self._rcv_base_path, 'industry_idx.json'), 'r')
		self._industry_index = json.load(industry_idx_file)
		industry_idx_file.close()

		industry_distribution_file = open(os.path.join(self._rcv_base_path, 'industry_distribution.json'), 'r')
		self._industry_distribution = json.load(industry_distribution_file)
		industry_distribution_file.close()

		country_idx_file = open(os.path.join(self._rcv_base_path, 'country_idx.json'), 'r')
		self._country_index = json.load(country_idx_file)
		country_idx_file.close()

		country_distribution_file = open(os.path.join(self._rcv_base_path, 'country_distribution.json'), 'r')
		self._country_distribution = json.load(country_distribution_file)
		country_distribution_file.close()

	def topic_list_for_doc_id(self, doc_id):
		return self._doc_index[doc_id]['topics']

	def industry_list_for_doc_id(self, doc_id):
		return self._doc_index[doc_id]['industries']

	def country_list_for_doc_id(self, doc_id):
		return self._doc_index[doc_id]['countries']

	def add_doc(self, doc_id, topic_list, country_list, industry_list, doc_path):
		self._doc_index[doc_id]['topics'].extend(topic_list)
		self._doc_index[doc_id]['industries'].extend(industry_list)
		self._doc_index[doc_id]['countries'].extend(country_list)
		self._doc_index[doc_id]['path'].append(doc_path)

		for topic in topic_list:
			self._topic_index[topic].append(doc_id)

		for country in country_list:
			self._country_index[country].append(doc_id)

		for industry in industry_list:
			self._industry_index[industry].append(doc_id)

	def store_index(self):
		print('Storing Doc Index...')
		doc_idx_file = open(os.path.join(self._rcv_base_path, 'doc_idx.json'), 'w')
		json.dump(self._doc_index, doc_idx_file)
		doc_idx_file.close()

		print('Storing Topic Index...')
		topic_idx_file = open(os.path.join(self._rcv_base_path, 'topic_idx.json'), 'w')
		json.dump(self._topic_index, topic_idx_file)
		topic_idx_file.close()

		print('Storing Industry Index...')
		industry_idx_file = open(os.path.join(self._rcv_base_path, 'industry_idx.json'), 'w')
		json.dump(self._industry_index, industry_idx_file)
		industry_idx_file.close()

		print('Storing Country Index...')
		country_idx_file = open(os.path.join(self._rcv_base_path, 'country_idx.json'), 'w')
		json.dump(self._country_index, country_idx_file)
		country_idx_file.close()

	def print_all(self):
		print('DOC INDEX:', self._doc_index)
		print('TOPIC INDEX:', self._topic_index)
		print('COUNTRY INDEX:', self._country_index)

def create_rcv1_index(rcv_path, verbose=False):
	rcv_cds = [os.path.join(rcv_path, 'cd1'), os.path.join(rcv_path, 'cd2')]

	rcv1_index = RCV1Index(rcv_path)

	for p in rcv_cds:
		os.chdir(p)
		files = [f for f in glob.glob('*.zip') if f[0].isdigit()]
		for f in files:
			curr_zip = os.path.join(p, f)
			_vprint('>    Reading zipfile: %s...' % (curr_zip,), verbose)
			with ZipFile(curr_zip, 'r') as zipzap:
				for fname in zipzap.namelist():
					the_file = zipzap.open(fname, 'rU')
					xmldoc = ET.parse(the_file)
					doc_id = None
					for newsitem in xmldoc.getroot().iter('newsitem'):
						doc_id = newsitem.attrib['itemid']
						class_dict = collections.defaultdict(list)
						for codes in newsitem.findall('./metadata/codes'):
							key = codes.attrib['class'].split(':')[1]
							for code in codes.iter('code'):
								class_dict[key].append(code.attrib['code'])
					rcv1_index.add_doc(doc_id=doc_id, topic_list=class_dict['topics'], country_list=class_dict['countries'], industry_list=class_dict['industries'], doc_path=os.path.join(curr_zip, fname))
	rcv1_index.store_index()

def _vprint(text, verbose):
	if (verbose):
		print(text)

if (__name__ == '__main__'):
	if (len(sys.argv) > 1):
		create_rcv1_index(sys.argv[1], verbose=True)
	else:
		sys.stderr.write('You need to provide the path to the rcv1 base folder!')
