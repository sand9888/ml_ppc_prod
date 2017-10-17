import pandas as pd
import os
import re

# reading datasets
root_dir = os.path.abspath('../ayur')

file_report = []

for filename in os.listdir(root_dir + '/csv'):
	if filename.endswith('main1.csv'):  # reading first part of basic dataset
		main1 = all_users1 = pd.read_csv(os.path.join(root_dir + '/csv', filename), names=['id', 'sessions', 'sess_duration', 'bounce_rate', 'revenue', 'transactions', 'conv_rate'], skiprows=6, header=0, dtype={'id':str})
	elif filename.endswith('main2.csv'):  # reading second part of basic dataset
		main2 = all_users1 = pd.read_csv(os.path.join(root_dir + '/csv', filename), names=['id', 'sessions', 'sess_duration', 'bounce_rate', 'revenue', 'transactions', 'conv_rate'], skiprows=6, header=0, dtype={'id':str})
	else:
		file_report.append(re.split('^Analytics_ayur_', filename)[1].split('.csv')[0]) # creating list of extra datasets

# creating basic dataset
all_users = main1.append(main2).reset_index()
row_number = all_users.shape[0]

# merging basic dataset and additional ones
for name in file_report:
	df_common = pd.read_csv(os.path.join(root_dir + '/csv', 'Analytics_ayur_' + name + '.csv'),
							names=['id', 'sessions', 'sess_duration', 'bounce_rate', 'revenue', 'transactions', 'conv_rate'],
							skiprows=6, header=0, dtype={'id': str})
	df_common[name] = 1
	df_common = df_common[['id', name]]
	all_users = pd.merge(all_users, df_common, how='left', on=['id'])

# some preprocessing
all_users['bounce_rate'] = all_users['bounce_rate'].str.replace(',', '.').str.replace('%', '')  # get rid of '%' sign
all_users['conv_rate'] = all_users['conv_rate'].str.replace(',', '.').str.replace('%', '')  # get rid of '%' sign
all_users = all_users[~all_users.sess_duration.str.contains('<')]  # get rid of '<' sign
all_users['sess_duration'] = pd.to_timedelta(all_users['sess_duration']).dt.total_seconds()  # turn initial time into seconds
all_users.drop(['revenue', 'transactions'], axis=1, inplace=True)
all_users.fillna(0, axis=1, inplace=True)

# saving to csv
all_users.to_csv('ayur_all_users.csv')
print (all_users.head(5))
