__author__ = 'thomas'

from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit


def split_data_raw(dataset, train_data_idx, train_labels_idx, test_data_idx, test_labels_idx, unlabelled_data_idx, random_state):
	X_train = dataset[train_data_idx]
	y_train = dataset[train_labels_idx]

	if (test_data_idx == -1 and unlabelled_data_idx == -1): # Going to be the RCV1 Dataset, plenty of data, so the splits should be reasonably sized
		# Do a 0.8 (train & unlabelled) vs. 0.2 (test) split
		try:
			split = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.2, random_state=random_state)
		except ValueError: # happens if not enough labels are available for a stratified split
			split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.2, random_state=random_state)

		for train_idx, test_idx in split:
			# Loop over the stuff, because lists don't do fancy indexing
			X_train_raw = []
			for idx in train_idx:
				X_train_raw.append(X_train[idx])
			X_test_raw = []
			for idx in test_idx:
				X_test_raw.append(X_train[idx])
			X_train, X_test = X_train_raw, X_test_raw
			y_train, y_test = y_train[train_idx], y_train[test_idx]

		# Do another 0.8 (unlabelled) vs. 0.2 (train) split
		try:
			split = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.2, random_state=random_state)
		except ValueError: # happens if not enough labels are available for a stratified split
			split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.2, random_state=random_state)

		for unlabelled_idx, train_idx in split:
			X_train_raw = []
			for idx in train_idx:
				X_train_raw.append(X_train[idx])
			X_unl_raw = []
			for idx in unlabelled_idx:
				X_unl_raw.append(X_train[idx])
			X_unlabelled, X_train = X_unl_raw, X_train_raw
			y_train = y_train[train_idx]

	elif (test_data_idx == -1 and unlabelled_data_idx != -1): # Twitter Datasets, not much labelled data
		# Do a 0.7 (train) vs. 0.3 (test)
		try:
			split = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=random_state)
		except ValueError: # happens if not enough labels are available for a stratified split
			split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.3, random_state=random_state)

		for train_idx, test_idx in split:
			X_train_raw = []
			for idx in train_idx:
				X_train_raw.append(X_train[idx])
			X_test_raw = []
			for idx in test_idx:
				X_test_raw.append(X_train[idx])
			X_train, X_test = X_train_raw, X_test_raw
			y_train, y_test = y_train[train_idx], y_train[test_idx]

		X_unlabelled = dataset[unlabelled_data_idx]

	elif (test_data_idx != -1 and unlabelled_data_idx == -1): # ApteMod & 20Newsgroups, splits should be appropriately sized
		# Do a 0.7 (unlabelled) vs. 0.3 (train)
		try:
			split = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=random_state)
		except ValueError: # happens if not enough labels are available for a stratified split
			split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.3, random_state=random_state)

		for unlabelled_idx, train_idx in split:
			X_train_raw = []
			for idx in train_idx:
				X_train_raw.append(X_train[idx])
			X_unl_raw = []
			for idx in unlabelled_idx:
				X_unl_raw.append(X_train[idx])
			X_unlabelled, X_train = X_unl_raw, X_train_raw
			y_train = y_train[train_idx]

		X_test, y_test = dataset[test_data_idx], dataset[test_labels_idx]

	elif (test_data_idx != -1 and unlabelled_data_idx != -1): # MovieReviews, all there, all happy, all fine, lets go to the Pub!
		X_test = dataset[test_data_idx]
		y_test = dataset[test_labels_idx]
		X_unlabelled = dataset[unlabelled_data_idx]

	return (X_train, y_train, X_test, y_test, X_unlabelled)


def split_data(dataset, train_data_idx, train_labels_idx, test_data_idx, test_labels_idx, unlabelled_data_idx, random_state):
	X_train = dataset[train_data_idx]
	y_train = dataset[train_labels_idx]

	if (test_data_idx == -1 and unlabelled_data_idx == -1): # Going to be the RCV1 Dataset, plenty of data, so the splits should be reasonably sized
		# Do a 0.8 (train & unlabelled) vs. 0.2 (test) split
		try:
			split = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.2, random_state=random_state)
		except ValueError: # happens if not enough labels are available for a stratified split
			split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.2, random_state=random_state)

		for train_idx, test_idx in split:
			X_train, X_test = X_train[train_idx], X_train[test_idx]
			y_train, y_test = y_train[train_idx], y_train[test_idx]

		# Do another 0.8 (unlabelled) vs. 0.2 (train) split
		try:
			split = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.2, random_state=random_state)
		except ValueError: # happens if not enough labels are available for a stratified split
			split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.2, random_state=random_state)

		for unlabelled_idx, train_idx in split:
			X_unlabelled, X_train = X_train[unlabelled_idx], X_train[train_idx]
			y_train = y_train[train_idx]

	elif (test_data_idx == -1 and unlabelled_data_idx != -1): # Twitter Datasets, not much labelled data
		# Do a 0.7 (train) vs. 0.3 (test)
		try:
			split = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=random_state)
		except ValueError: # happens if not enough labels are available for a stratified split
			split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.3, random_state=random_state)

		for train_idx, test_idx in split:
			X_train, X_test = X_train[train_idx], X_train[test_idx]
			y_train, y_test = y_train[train_idx], y_train[test_idx]

		X_unlabelled = dataset[unlabelled_data_idx]

	elif (test_data_idx != -1 and unlabelled_data_idx == -1): # ApteMod & 20Newsgroups, splits should be appropriately sized
		# Do a 0.7 (unlabelled) vs. 0.3 (train)
		try:
			split = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=random_state)
		except ValueError: # happens if not enough labels are available for a stratified split
			split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.3, random_state=random_state)

		for unlabelled_idx, train_idx in split:
			X_unlabelled, X_train = X_train[unlabelled_idx], X_train[train_idx]
			y_train = y_train[train_idx]

		X_test, y_test = dataset[test_data_idx], dataset[test_labels_idx]

	elif (test_data_idx != -1 and unlabelled_data_idx != -1): # MovieReviews, all there, all happy, all fine, lets go to the Pub!
		X_test = dataset[test_data_idx]
		y_test = dataset[test_labels_idx]
		X_unlabelled = dataset[unlabelled_data_idx]

	return (X_train, y_train, X_test, y_test, X_unlabelled)