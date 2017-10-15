def test_get_metric_report():
	expected =  [1,0,0,1,1,1,1,0]
	predicted = [1,1,1,0,1,1,1,1]
	from org.changyy.ml.performance_metrics import get_metric_report
	report, score = get_metric_report(expected, predicted, 'weighted')
	assert round(score, 2) == 0.42
	assert len(report) > 10

def test_get_metric_report_iris_case():
	print()
	import pandas as pd
	from org.changyy.ml.resource import get_iris_sample_data
	dataset = get_iris_sample_data()

	from org.changyy.ml.feature_engineering import data_numeric_handler_process
	_, handled_dataset = data_numeric_handler_process(dataset.copy(), skip_columns=['class'])

	y = handled_dataset['class']
	train = handled_dataset.drop(['class'], axis=1) 
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=0)

	from sklearn.multiclass import OneVsOneClassifier
	from sklearn.svm import LinearSVC
	classifier = OneVsOneClassifier(LinearSVC(random_state = 0))

	classifier.fit(X_train, y_train)
	expected = y_test
	predicted = classifier.predict(X_test)

	from org.changyy.ml.performance_metrics import get_metric_report
	report, score = get_metric_report(expected, predicted)
	assert round(score, 2) == round(classifier.score(X_test, y_test), 2)
	assert len(report) > 10

def test_get_metric_value():
	expected =  [1,0,0,1,1,1,1,0]
	predicted = [1,1,1,0,1,1,1,1]
	from org.changyy.ml.performance_metrics import get_metric_value
	precision_score, recall_score, average_precision_score = get_metric_value(expected, predicted, 'macro')
	#print(precision_score, recall_score, average_precision_score)
	assert round(0.285714285714, 5) == round(precision_score, 5) 
	assert round(0.4, 2) == round(recall_score, 2)
	assert round( 0.582142857143, 5) == round(average_precision_score, 5)

def test_get_roc_report():
	expected =  [1,0,0,1,1,1,1,0]
	predicted = [1,1,1,0,1,1,1,1]
	from org.changyy.ml.performance_metrics import get_metric_roc

	f, t, roc = get_metric_roc(expected, predicted)
	print(roc)
	assert round(0.4, 2) == round(roc, 2)
