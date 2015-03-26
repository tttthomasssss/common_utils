__author__ = 'thk22'
import numpy as np


def margin_of_confidence_ranking(nb_model, Z, ordering='asc', with_evidence=False):
	# Classify the stuff given the current model
	probas = nb_model.predict_proba(Z)

	# Rank the labels for the probabilistically labelled instances
	sorted_idx = np.argsort(-probas, axis=1) # rowwise sort, -probas so the sort is descending rather than ascending

	if (with_evidence): # evidence makes the distinction between MSU/LSU
		# Get the feature (a.k.a. attribute) sets for the most-likely class (mlc) and the second most-likely class (smlc)
		evidence = nb_model.estimate_evidence(sorted_idx, Z)

		# Rank the Instances
		to_label_idx = np.argsort(evidence) if ordering == 'asc' else np.argsort(-evidence)
	else: # Only uses the margin of confidence as criterion, independent of MSU/LSU
		# Margin of confidence, subtract most confident label prediction from 2nd most confident label prediction
		conf_margins = probas[np.arange(sorted_idx.shape[0]), sorted_idx[:, 0]] - probas[np.arange(sorted_idx.shape[0]), sorted_idx[:, 1]]

		# Rank the Instances
		to_label_idx = np.argsort(conf_margins) if ordering == 'asc' else np.argsort(-conf_margins)

	return to_label_idx