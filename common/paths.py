__author__ = 'thomas'
import os
import socket

PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))

__BASEPATH__ = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab')
}

__DATASET_PATH__ = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/EpicDataShelf/tag-lab'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_datasets/')
}

__OUT_PATH__ = {
	'Thomas-Kobers-MacBook.local': os.path.expanduser('~/DevSandbox/InfiniteSandbox/tag-lab/_results'),
	'm011437.inf.susx.ac.uk': os.path.expanduser('~/DevSandbox/InfiniteSandbox/_results')
}


def get_base_path():
	return __BASEPATH__.get(socket.gethostname(), '/mnt/lustre/scratch/inf/thk22/code') # Fallback on lustre path on cluster


def get_dataset_path():
	return __DATASET_PATH__.get(socket.gethostname(), '/mnt/lustre/scratch/inf/thk22/_datasets') # Fallback on lustre path on cluster


def get_out_path():
	return __OUT_PATH__.get(socket.gethostname(), '/mnt/lustre/scratch/inf/thk22/_results') # Fallback on lustre path on cluster