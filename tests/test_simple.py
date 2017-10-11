# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

def test_success():
	assert True

def test_feature_engineering():
	import pandas as pd
	# https://archive.ics.uci.edu/ml/datasets/Iris
	dataset = pd.read_csv(
		'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
		# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names
		names = [ 'sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
	)
	print()
	print("dataset:", dataset.shape)
	assert dataset.shape[0] > 100
	print("columns:\n", dataset.columns)

	from org.changyy.ml.helper import data_numeric_handler_process

	print("raw:\n\n", dataset.head(3))
	_, handled_dataset = data_numeric_handler_process(dataset.copy(), skip_columns=['class'])
	print("Encoding:\n\n", handled_dataset.head(3))

	assert set(handled_dataset['class'].unique()) == set(dataset['class'].unique())
	#print(handled_dataset['class'].unique())
	for column in handled_dataset.columns:
		assert len(handled_dataset[column].unique()) == len(dataset[column].unique())
		if column != 'class':
			assert set(handled_dataset[column].unique()) != set(dataset[column].unique())

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

	from sklearn import metrics
	print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))

	score = classifier.score(X_test, y_test)
	print("Score:", score)
	assert round(score, 10) == round(0.84210526315, 10)

	#_, handled_dataset = data_numeric_handler_process(dataset.copy(), skip_columns=['class'], onehotencode_columns=['sepal length in cm'], lookup_table={})
	_, handled_dataset = data_numeric_handler_process(dataset.copy(), skip_columns=['class'], onehotencode_columns=list(set(dataset.columns)), lookup_table={})
	#print(handled_dataset.columns)
	assert len(handled_dataset.columns) == 131
	y = handled_dataset['class']
	train = handled_dataset.drop(list(set(dataset.columns)), axis=1) 
	#print(train.columns)
	assert len(train.columns) == 126

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=0)

	from sklearn.multiclass import OneVsOneClassifier
	from sklearn.svm import LinearSVC
	classifier = OneVsOneClassifier(LinearSVC(random_state = 0))

	classifier.fit(X_train, y_train)
	expected = y_test
	predicted = classifier.predict(X_test)

	from sklearn import metrics
	print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))

	score = classifier.score(X_test, y_test)
	print("Score:", score)
	assert round(score, 10) == round(1.0, 10)

