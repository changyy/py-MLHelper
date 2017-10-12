def get_iris_sample_data():
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	tmp_file_name = 'ml-get_iris_sample_data.log'

	import os
	import tempfile
	tmp_file = os.path.join(tempfile.gettempdir(), tmp_file_name)
	if not os.path.exists(tmp_file) or os.path.getsize(tmp_file) == 0:
		import urllib.request
		urllib.request.urlretrieve(url, tmp_file)

	import pandas as pd
	# https://archive.ics.uci.edu/ml/datasets/Iris
	dataset = pd.read_csv(
		url if not os.path.exists(tmp_file) or os.path.getsize(tmp_file) == 0 else tmp_file,
		# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names
		names = [ 'sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'class']
	)

	return dataset

def get_adult_sample_data():
	# https://archive.ics.uci.edu/ml/datasets/adult
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
	tmp_file_name = 'ml-get_adult_sample_data.log'

	import os
	import tempfile
	tmp_file = os.path.join(tempfile.gettempdir(), tmp_file_name)
	if not os.path.exists(tmp_file) or os.path.getsize(tmp_file) == 0:
		import urllib.request
		urllib.request.urlretrieve(url, tmp_file)

	import pandas as pd
	dataset = pd.read_csv(
		url if not os.path.exists(tmp_file) or os.path.getsize(tmp_file) == 0 else tmp_file,
		# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
		names = [ 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
	)

	return dataset

