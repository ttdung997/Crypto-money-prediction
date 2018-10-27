import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
import datetime
import sys
import io
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
from keras.preprocessing import sequence
from scipy import interp


EXPERT_SCORE = 0.6
TRAINING_FLAG = 0

# coin list
coin_list = ['bitcoin','ethereum','ripple','bitcoin-cash',
'eos','stellar','litecoin','tether','monero']

# build model LSTM function
def build_model(inputs, output_size, neurons, activ_func="linear",
				dropout=0.25, loss="mae", optimizer="adam"):
	model = Sequential()

	model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
	model.add(Dropout(dropout))
	model.add(Dense(units=output_size))
	model.add(Activation(activ_func))

	model.compile(loss=loss, optimizer=optimizer)

	return model

# load data to training
Dataframe = pd.read_csv("data/final.csv")
ColumnList = list(Dataframe)
Data = Dataframe.values

# preprocessing data 
for i in range(0, len(Data)):
	for j in range(0,len(ColumnList)):
		maxIn = Dataframe[ColumnList[j]].max()
		minIn = Dataframe[ColumnList[j]].min()
		if maxIn == minIn:
			Data[i][j] = 0
		else:
			Data[i][j] = float(Data[i][j] - minIn)/float(maxIn-minIn)

Data = np.asarray(Data)

# split to train and test set
total_train_set = [x for x in Data[:352]]
total_test_set  = [x for x in Data[353:]]


window_len = 50

#training
for coin in coin_list:
	# fix data
	train_label = [x[ColumnList.index(coin+"_close")] for x in total_train_set]
	test_label =  [x[ColumnList.index(coin+"_close")] for x in total_test_set]
   
	train_set=[(x[:ColumnList.index(coin+"_close")]+x[(ColumnList.index(coin+"_close")):]) for x in total_train_set]
	test_set=[(x[:ColumnList.index(coin+"_close")]+x[(ColumnList.index(coin+"_close")):]) for x in total_test_set]

	LSTM_training_inputs = []

	for i in range(len(train_set)-window_len):
		temp_set = train_set[i:(i+window_len)].copy()
		LSTM_training_inputs.append(temp_set)

	LSTM_test_inputs = []
	for i in range(len(test_set)-window_len):
		temp_set = test_set[i:(i+window_len)].copy()
		LSTM_test_inputs.append(temp_set)

	LSTM_last_input=temp_set


	LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
	LSTM_training_inputs = np.array(LSTM_training_inputs)

	LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
	LSTM_test_inputs = np.array(LSTM_test_inputs)
	
	LSTM_train_outputs = np.array(train_label)[window_len:]
	LSTM_test_outputs = np.array(test_label)

  
	maxout =Dataframe[coin+"_close"].max()
	minout =Dataframe[coin+"_close"].min()

	model_output = "model/"+coin+"_model.json"
	weight_output = "model/"+coin+"_model.h5"

	# train model
	if TRAINING_FLAG >0:
		model =build_model(LSTM_training_inputs, output_size=1, neurons = 100)
		model.fit(LSTM_training_inputs, LSTM_train_outputs, 
								epochs=10, batch_size=1, verbose=1, shuffle=True)
		print("predict on change")
		print(model.predict(LSTM_training_inputs))
		model_json = model.to_json()
		with open(model_output, "w") as json_file:
				json_file.write(model_json)
				# serialize weights to HDF5
				model.save_weights(weight_output)

	break
# load model
json_file = open(model_output, 'r')
loaded_model_json = json_file.read()
json_file.close()
load_model = model_from_json(loaded_model_json)
load_model.load_weights(weight_output)

# get data to visualization
labelArr = []
predictArr = []
err =[]
predict = load_model.predict(LSTM_test_inputs)

for i in range(0,len(test_label[window_len:])):
  label = test_label[window_len:][i]*(maxout-minout)+minout
  labelArr.append(label)
  pre = (predict[i]*(maxout-minout)+minout) * EXPERT_SCORE
  predictArr.append(pre)
  err.append(abs(label - pre))

from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
# draw plot

# tick every 5th easter
rule = rrulewrapper(YEARLY, byweekday=1, interval=5)
loc = RRuleLocator(rule)
formatter = DateFormatter('%m/%d/%y')
date1 = datetime.date(2018, 9,22)
date2 = datetime.date(2018, 10, 30)
delta = datetime.timedelta(days=1)

dates = drange(date1, date2, delta)

predict_date=pd.read_csv('data/date.csv')
print(predict_date)
fig, ax1 = plt.subplots(1,1)


ax1.plot(dates,
		 (labelArr),
		  label='Actual')
ax1.plot(dates,
		 (predictArr),
		  label='Predicted')

# ax1.plot(labelArr, label='Actual')
# ax1.plot(predict_date.tail(38)['date'].astype(datetime.datetime),
# 		 (predictArr),
		  # label='Predicted')
ax1.annotate('MAE: %.4f'%np.mean(err), 
			 xy=(0.75, 0.9),  xycoords='axes fraction',
			xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})

ax1.xaxis.set_major_locator(loc)
ax1.xaxis.set_major_formatter(formatter)
ax1.xaxis.set_tick_params(rotation=10, labelsize=10)

plt.show()


