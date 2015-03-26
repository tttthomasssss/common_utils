__author__ = 'thk22'
from datetime import datetime

# incl timestamp: format='%d%m%Y_%H%M%S'
def timestamped_foldername(format='%d%m%Y'):
	return datetime.now().strftime(format)