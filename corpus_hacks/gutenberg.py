__author__ = 'thk22'
from glob import iglob
import csv
import logging
import os
import re
import sys

from common import paths


def create_csv(base_path, lowercase=True):
	count = 0
	with open(os.path.join(paths.get_dataset_path(), 'gutenberg', 'gutenberg_lowercase-{}.tsv'.format(lowercase)), 'w') as out_file:
		csv_writer = csv.writer(out_file, delimiter='\t')
		for idx, f in enumerate(iglob(os.path.join(base_path, 'www.gutenberg.lib.md.us', '**', '*.txt'), recursive=True)):

			# Files that contain a hyphen in their name have some shitty ISO encoding and fuck up the reading, hence we ignore them
			# which is fine because there always is an ASCII encoded file anyway and hence we prevent double counts as well :D
			_, tail = os.path.split(f)
			if ('-' not in tail):
				logging.info('[{}] Processing {}...'.format(idx, f))

				contents = None

				try: # FUCK ALL ENCODINGS!!!
					with open(f, 'r', encoding='utf-8') as in_file:
						contents = in_file.read()

						# Replace all linebreaks, tabs and multiple consecutive spaces with a single space
						contents = contents.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
						contents = re.sub(r'\s\s+', ' ', contents)
				except UnicodeDecodeError as ex:
					logging.error('Encoding fuckup, trying another one! - {}'.format(ex))
					with open(f, 'r', encoding='ISO-8859-1') as in_file:
						contents = in_file.read()

						# Replace all linebreaks, tabs and multiple consecutive spaces with a single space
						contents = contents.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
						contents = re.sub(r'\s\s+', ' ', contents)

				# Sanity!
				if (contents is None or contents == ''):
					logging.error('SHITS GONE WRONG!!!! CONTENTS IS EMPTY!!!! IDX={} FILE={}'.format(idx, f))
					sys.exit(666)
				else: # Write shit
					csv_writer.writerow([contents.lower() if lowercase else contents])
			else:
				logging.info('[{}] Skipping {}...'.format(idx, f))

		count = idx

	logging.info('Successfully created Gutenberg corpus ({} files)!'.format(count))


if (__name__ == '__main__'):
	logging.basicConfig(format='%(asctime)s %(message)s', datefmt='[%m/%d/%Y %I:%M:%S] %p')
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.DEBUG)
	root_logger.addHandler(logging.StreamHandler(sys.stdout))

	create_csv(os.path.join(paths.get_dataset_path(), 'gutenberg'))