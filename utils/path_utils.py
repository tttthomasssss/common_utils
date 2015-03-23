__author__ = 'thk22'
from datetime import datetime


def timestamped_foldername(format='%d%m%Y_%H%M%S'):
	return datetime.now().strftime(format)