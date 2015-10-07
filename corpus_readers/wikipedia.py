__author__ = 'thomas'
import csv
import sys

from nltk import word_tokenize


class WikipediaReader(object):
	def __init__(self, file_path, tokeniser=word_tokenize, lowercase=True):
		csv.field_size_limit(sys.maxsize)
		self.file_path_ = file_path
		self.tokeniser_ = tokeniser
		self.lowercase_fn_ = lambda t: t.lower() if lowercase else lambda t: t

	def _preprocess(self, text):
		return self.tokeniser_(self.lowercase_fn_(text))

	def __iter__(self):
		with open(self.file_path_, mode='r', encoding='utf-8') as csv_file:
			csv_reader = csv.reader(csv_file)

			for line in csv_reader:
				yield self._preprocess(line[1])


class Wikipedia(object):
	def __init__(self):
		self.vocab_ = None
		self.vocab_size_ = 0
		self.index_ = None
		self.inverted_index_ = None
		self.token_counts_ = None