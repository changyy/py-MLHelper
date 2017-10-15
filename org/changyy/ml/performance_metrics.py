def get_metric_report(expected, predicted,f1_score_method='weighted', only_final_score=False):
	# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
	from sklearn import metrics
	if only_final_score:
		 return metrics.f1_score(expected, predicted, average=f1_score_method)
	return metrics.classification_report(expected, predicted), metrics.f1_score(expected, predicted, average=f1_score_method)

def get_metric_value(expected, predicted, average_score_method='macro', only_final_score=False):
	from sklearn.metrics import average_precision_score, precision_score, recall_score
	if only_final_score:
		return average_precision_score(expected, predicted, average=average_score_method)
	return precision_score(expected, predicted, average=average_score_method), recall_score(expected, predicted, average=average_score_method), average_precision_score(expected, predicted, average=average_score_method)

def get_metric_roc(expected, predicted):
	# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, _ = roc_curve(expected, predicted)
	roc_auc = auc(fpr, tpr)
	return fpr, tpr, roc_auc
	# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
