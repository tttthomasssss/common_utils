__author__ = 'thomas'
from common import dataset_utils


def read_scws():
	data = dataset_utils.fetch_scws_dataset()
	print 'Done!'


if (__name__ == '__main__'):
	read_scws()