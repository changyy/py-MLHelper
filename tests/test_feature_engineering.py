def test_feature_engineering_data_numeric_handler_process():
	print()
	import pandas as pd
	from org.changyy.ml.resource import get_iris_sample_data
	dataset = get_iris_sample_data()
	#print("dataset:", dataset.shape)
	#assert dataset.shape[0] > 100
	#print("columns:\n", dataset.columns)

	from org.changyy.ml.feature_engineering import data_numeric_handler_process

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


def test_feature_engineering_data_functional_handler_process():
	print()
	import pandas as pd
	from org.changyy.ml.resource import get_adult_sample_data
	dataset = get_adult_sample_data()

	print(dataset.head(3))
	print(dataset.columns)

	from org.changyy.ml.feature_engineering import data_functional_handler_process

	def age_handler(panda_obj):
		#panda_obj['age'] = panda_obj['age'].astype(int)
		last_index = -1
		for index, value in enumerate(range(80, 10, -10)):
			panda_obj.loc[ panda_obj['age'] >= value, 'age'] = index
			last_index = index
		panda_obj.loc[ panda_obj['age'] > last_index, 'age'] = last_index + 1

	#print("age raw:", dataset['age'].unique())
	data_functional_handler_process(dataset, {'age': age_handler})
	#print("age encoding:", dataset['age'].unique())

	from org.changyy.ml.feature_engineering import data_numeric_handler_process
	#_, handled_dataset = data_numeric_handler_process(dataset.copy(), skip_columns=['age','salary'], onehotencode_columns=list(set(dataset.columns)), lookup_table={})
	onehotencode_columns = ['relationship','race', 'occupation', 'sex']
	remove_columns = ['fnlwgt', 'hours-per-week']
	#onehotencode_columns = []
	_, handled_dataset = data_numeric_handler_process(dataset.copy(), skip_columns=['age','salary'], onehotencode_columns=onehotencode_columns, lookup_table={})
	#print(handled_dataset.columns)
	y = handled_dataset['salary']
	train = handled_dataset.drop(['salary','education'], axis=1) 
	if len(onehotencode_columns) > 0:
		train = train.drop(onehotencode_columns, axis=1) 
	if len(remove_columns) > 0:
		train = train.drop(remove_columns, axis=1)
	#print(train.corr())
	#print(train.columns)

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

	assert round(score, 10) == round(0.819678172215, 10)
