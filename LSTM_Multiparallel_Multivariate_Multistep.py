# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
import numpy as np

def split_sequences(sequences, n_steps_in, n_steps_out):
 X, y = list(), list()
 for i in range(len(sequences)):
  end_ix = i + n_steps_in
  out_end_ix = end_ix + n_steps_out
  if out_end_ix > len(sequences):
    break
  seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
  X.append(seq_x)
  y.append(seq_y)
 return array(X), array(y)

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
n_steps_in, n_steps_out = 9, 9
# data_train = dataset_new[ 0:550 , : ]
# data_test = dataset_new[ 550:778 , : ]

X, y = split_sequences(dataset_new, n_steps_in, n_steps_out)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
n_features = X.shape[2]

# X_train, y_train = split_sequences(data_train, n_steps_in, n_steps_out)
# X_test, y_test = split_sequences(data_test, n_steps_in, n_steps_out)
# n_features = X.shape[2]

print(y_test.shape)


model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=300, verbose=0)
model.save('LSTM.h5')

loaded_model = load_model('LSTM.h5')

yhat = loaded_model.predict(X_test, verbose=0)

print(yhat.shape)

for i in range(153):
  dist = np.linalg.norm(yhat[i]-y_test[i])
  phantram = abs((yhat[i]-y_test[i])/y_test[i])
  print(phantram)

import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu mảng 2 chiều
array1 = yhat.reshape((1377,9))
array2 = y_test.reshape((1377,9))

plt.plot(array1[ 100:180 , 0 ], label='temp_predict')
plt.plot(array2[ 100:180 , 0 ], label='temp_real')

dist = np.linalg.norm(array1[ : , 0 ]-array2[ : , 0 ])
print(dist)

# Đặt tiêu đề và chú thích
# plt.title('Comparison of Row {}'.format(row_index))
plt.xlabel('Column Index')
plt.ylabel('Value')
plt.legend()

# Hiển thị biểu đồ
plt.grid(True)
plt.show()