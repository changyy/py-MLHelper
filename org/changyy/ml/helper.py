def fill_value_via_statistics_handler(panda_obj, column_name='age', max_value=120):
	import pandas
	import numpy
	if type(panda_obj) is not pandas.core.frame.DataFrame:
		return panda_obj
	panda_obj.loc[ panda_obj[column_name][panda_obj[column_name] > max_value ].index , column_name ] = numpy.nan

	value_avg = panda_obj[column_name].mean()
	value_std = panda_obj[column_name].std()
	value_null_count = panda_obj[column_name].isnull().sum()

	numpy.random.seed(0)
	value_null_random_list = numpy.random.randint(value_avg - value_std, value_avg + value_std, size=value_null_count)
	panda_obj.loc[ panda_obj[column_name][numpy.isnan(panda_obj[column_name])].index , column_name ] = value_null_random_list

	panda_obj[column_name] = panda_obj[column_name].astype(int)

	return panda_obj

def data_functional_handler_process(panda_obj,column_handler={}):
	import pandas
	import numpy
	if type(panda_obj) is not pandas.core.frame.DataFrame:
		return panda_obj
	if type(column_handler) is not dict:
		return panda_obj

	for column in list(set(panda_obj.columns)):
		if column in column_handler:
			column_handler[column](panda_obj)

	return panda_obj

def data_numeric_handler_process(panda_obj,skip_columns=[],target_columns=[],onehotencode_columns=[],lookup_table={}):
	import pandas
	import numpy
	from sklearn.preprocessing import OneHotEncoder
	if type(panda_obj) is not pandas.core.frame.DataFrame:
		return lookup_table, panda_obj
	if target_columns is None or len(target_columns) == 0:
		target_columns = list(set(panda_obj.columns))
	if skip_columns is not None and len(skip_columns) > 0:
		for column in skip_columns:
			if column in target_columns:
				target_columns.remove(column)

	if onehotencode_columns is not None and len(onehotencode_columns) > 0:
		for column in onehotencode_columns:
			if column in target_columns:
				target_columns.remove(column)

	for column in target_columns:
		if column not in lookup_table:
			lookup_table[column] = dict((value,index+1) for index, value in enumerate(panda_obj[column].unique()))
		else:
			max_index = len(lookup_table[column]) + 1 
			for value in panda_obj[column].unique():
				if value not in lookup_table[column]:
					lookup_table[column][value] = max_index
					max_index = max_index + 1 
		panda_obj[column] = panda_obj[column].map(lookup_table[column])
		panda_obj[column] = panda_obj[column].fillna(0)
		panda_obj[column] = panda_obj[column].astype(int)

	for column in onehotencode_columns:
		if column not in lookup_table:
			values = list(set(panda_obj[column].unique()))
			lookup_table[column] = dict((value,index) for index, value in enumerate(panda_obj[column].unique()))
		panda_obj[column] = panda_obj[column].map(lookup_table[column])
		#panda_obj[column] = panda_obj[column].fillna(0)
		panda_obj[column] = panda_obj[column].astype(int)
	
		onehot_encoder = OneHotEncoder(sparse=False,n_values=len(lookup_table[column]))
		onehot_result = onehot_encoder.fit_transform(panda_obj[column].values.reshape(panda_obj[column].shape[0], 1))

		for index in range(len(lookup_table[column])):
			panda_obj['OneHotEncode-'+column+'-'+str(index)] = onehot_result[:,index]

	return lookup_table, panda_obj
