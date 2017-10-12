def test_resource_get_iris_sample_data():
	from org.changyy.ml.resource import get_iris_sample_data
	dataset = get_iris_sample_data()
	assert dataset.shape[0] > 100

def test_resource_get_adult_sample_data():
	from org.changyy.ml.resource import get_adult_sample_data
	dataset = get_adult_sample_data()
	assert dataset.shape[0] > 100
