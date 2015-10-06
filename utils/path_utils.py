__author__ = 'thk22'
from datetime import datetime
import os

# incl timestamp: format='%d%m%Y_%H%M%S'
def timestamped_foldername(format='%d%m%Y'):
	return datetime.now().strftime(format)


def check_all_exist(dataset_path, filenames):
	exists = True
	for fname in filenames:
		exists = exists and os.path.exists(os.path.join(dataset_path, fname))

	return exists