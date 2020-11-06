import math
from sklearn.externals import joblib
from keras.models import load_model
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time, os
from pandas import read_csv
from datetime import datetime
import numpy as np
import pandas as pd
from Data_interaction import DataInteraction 
import warnings
warnings.filterwarnings("ignore")

# @dev predict hum or tem variable in next step
class Prediction():
	def __init__(self):
		self.DI = DataInteraction()

	def get_res(self):
		self.train_model()
		input = self._getTrainData()		
		res =  self.predict(input)
		# print('predicted 15min outdoor indicators:{}\n'.format(res[0,:-1]))
		return res[0,:-1]

	def train_model(self):
		time_str = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
		year_str = time_str.split('-')[0][-2:]
		month_str = time_str.split('-')[1]
		day_str = time_str.split('-')[2]
		hour_str = time_str.split('-')[3]
		min_str = time_str.split('-')[4]
		# day in 2, 9, 16,23 update prediction model
		if (int(day_str) in [2,9,16,23,26]) and (int(hour_str) == 23) and (int(min_str)<=40):			
			self._train_prediction()

# @dev transform data into 5-min interval	
	def _getTrainData(self):
		raw = self.DI.getDf(1500)
		data = raw.iloc[:,[0,1,3,4,6,8,9,24]]
		data.columns = ['timestamp', 'date','To','Ho','So','Ti','Hi','CTL_1']
		data.loc[:,'timestamp'] = data.loc[:,'date'].apply(lambda x: x.timestamp())
		input = data.loc[:,['timestamp', 'To', 'Ho','So']].values.tolist()
		timestamp_last = input[0][0]
		# print(input)
		output = []
		n_row = 0
		for i in range(12):
			while(1):
				if (input[n_row][0] <= timestamp_last-900*i):
					output.append(input[i][1:])
					break
				n_row += 1		
		output = np.array(output)
		# print(output)
		return output

	def _series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
		n_vars = 1 if type(data) is list else data.shape[1]
		df = pd.DataFrame(data)
		cols, names = list(), list()
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(df.shift(-i))
			if i == 0:
				names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
			else:
				names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		# put it all together
		agg = concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			agg.dropna(inplace=True)
		return agg
	# **
	# * @dev train SVR prediction model and use it to predict
	# * @pram col The column of variable to be predicted
	# * @pram data The dataset for training and prediction
	# * @return linear_svr_y_predict The outcome of prediction
	#
	def _train_prediction(self):
		time_str = time.strftime('%Y-%m-%d',time.localtime(time.time()))
		year_str = time_str.split('-')[0][-2:]
		month_str = time_str.split('-')[1]
		day_str = time_str.split('-')[2]
		filename_data_ = ""
		dataset = pd.DataFrame()
		for month in range(int(month_str), 0, -1):
			for day in range(31, 0, -1):
				month = '0' + str(month) if len(str(month)) == 1 else str(month)
				day = '0' + str(day) if len(str(day)) == 1 else str(day)
				filename_data = '15min_' + year_str + month + day +'.csv' 
				# print(filename_data)
				try:	
					dataset = read_csv('../../data/env/'+ filename_data,  index_col=0)
					filename_data_ = filename_data
					break
				except:
					pass
			if len(dataset) > 0:
				break
		
		dataset = dataset[['To','Ho','So']].iloc[0:10000,:]

		values = dataset.values
		values = list(values)
		values.reverse()
		values = np.array(values)
		values = values.astype('float32')

		scaler = MinMaxScaler(feature_range=(0, 1))
		scaler_filename = '../../data/scaler/scaler.save'
		
		scaled = scaler.fit_transform(values)
		joblib.dump(scaler, scaler_filename)

		reframed = self._series_to_supervised(scaled, n_in = 12, n_out = 1 )
		# split into train and test sets
		values = reframed.values[:,:]

		n_train = int(0.9 * len(values))
		train = values[:n_train, :]
		test = values[n_train:, :]
		# # split into input and outputs
		train_X, train_y = train[:, :-3], train[:, -3:]
		test_X, test_y = test[:, :-3], test[:, -3:]

		# # reshape input to be 3D [samples, timesteps, features]
		train_X = train_X.reshape((train_X.shape[0], 12, 3))
		test_X = test_X.reshape((test_X.shape[0], 12, 3)) 
		# design network
		model = Sequential()
		model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
		model.add(Dense(3))
		model.compile(loss='mae', optimizer='adam')
		# fit network
		history = model.fit(train_X, train_y, epochs=300, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
		# print(filename_data)
		output_filename = 'lstm_'+str(filename_data_.split('.')[0])+'.h5'
		# print(output_filename)
		model.save("../../data/lstm_model/" + output_filename)
		print('lstm model saved in file {}'.format(output_filename))
	
	def predict(self, input):
		time_str = time.strftime('%Y-%m-%d',time.localtime(time.time()))
		year_str = time_str.split('-')[0][-2:]
		month_str = time_str.split('-')[1]
		day_str = time_str.split('-')[2]
		inv_yhat = np.zeros(1)
		for month in range(int(month_str), 0, -1):
			for day in range(31, 0, -1):
				month = '0' + str(month) if len(str(month)) == 1 else str(month)
				day = '0' + str(day) if len(str(day)) == 1 else str(day)
				filename_model = 'lstm_15min_' + year_str + month + day +'.h5'
				
				filename_data = '15min_' + year_str + month + day +'.csv'
				# print(filename_model, filename_data)
				if os.path.exists("../../data/lstm_model/"+filename_model):
					model = load_model("../../data/lstm_model/"+filename_model)
					# get scaler
					# dataset = read_csv('../../data/env/'+filename_data,  index_col=0)
					# dataset = dataset[['To','Ho','So']]
					# values = dataset.values
					# values = list(values)
					# values.reverse()
					# values = np.array(values)
					# values = values.astype('float32')
					# print(values)
					scaler = joblib.load("../../data/scaler/scaler.save")
					# scaler = MinMaxScaler(feature_range=(0, 1))
					# scaler.fit_transform(values)

					# input shape is n*36
					print(scaler)
					yhat = model.predict(scaler.transform(input).reshape(1,12,3))
					# print(yhat)
					inv_yhat = np.array(scaler.inverse_transform(yhat))
					break
			print(inv_yhat)
			if np.sum(abs(inv_yhat))>0 and np.sum(abs(inv_yhat[0]))>0:
				break 

		return inv_yhat

if __name__ == '__main__':
	Pre = Prediction()
	# print(Pre._getTrainData())
	Pre.get_res()