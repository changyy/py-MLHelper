def get_metric_report(expected, predicted,f1_score_method='weighted'):
	# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
	from sklearn import metrics
	return metrics.classification_report(expected, predicted), metrics.f1_score(expected, predicted, average=f1_score_method)

def get_metric_value(X_test, y_test, only_final_score=False):
	from sklearn.metrics import average_precision_score, precision_score, recall_score
	if only_final_score:
		return average_precision_score(y_test, X_test)
	return precision_score(y_test, X_test, average='macro'), recall_score(y_test, X_test, average='macro'), average_precision_score(y_test, X_test)

def get_roc_report(X_test, y_test):
	from sklearn.metrics import roc_curve, auc
