# multivariate output 1d cnn example
#https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
from numpy import array
from numpy import hstack
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
import numpy as np

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
#in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
#in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
#out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

dataset = np.loadtxt('/content/SensorData700.csv', delimiter=',')


in_seq1 = dataset[ : , 0]
in_seq2 = dataset[ : , 1]
in_seq3 = dataset[ : , 2]
in_seq4 = dataset[ : , 3]
in_seq5 = dataset[ : , 4]
in_seq6 = dataset[ : , 5]
in_seq7 = dataset[ : , 6]
in_seq8 = dataset[ : , 7]
in_seq9 = dataset[ : , 8]

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq6 = in_seq6.reshape((len(in_seq6), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
# horizontally stack columns
dataset_new = np.hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, in_seq6, in_seq7, in_seq8, in_seq9))
# choose a number of time steps
n_steps = 4
# convert into input/output
X, y = split_sequences(dataset_new, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=3000, verbose=0)
# model.save('multiparallel.h5')
# demonstrate prediction
#x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = array([[28.6,78.9,28.5,22.4,6.15,269,19,27,54], [28.2,80.2,28.4,22.3,6.14,263,18,26,52], [27.9,81.8,28.3,21.9,6.15,251,18,25,50], [27.7, 82.4, 28.2, 21.7, 61.4, 245, 17, 24, 49]])

x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
yreal = [27.5,83.5,28.1,21.5,6.13,238,16,23,47]
dist = np.linalg.norm(yreal-yhat)
print(yhat, yreal)
print(dist)

from tensorflow.keras.models import load_model
import numpy as np

loaded_model = load_model('multiparallel.h5')

x_input = array([[25.63,60.55,1011.73], [25.64,60.5,1011.73], [25.65,60.47,1011.73]])
x_input = x_input.reshape((1, n_steps, n_features))

yhat = loaded_model.predict(x_input, verbose=0)
print(yhat)