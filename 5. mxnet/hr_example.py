import mxnet as mx
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix



# Read data and basic statistics
hr_data = pd.read_csv("data/hr.csv")

print hr_data.head()
print hr_data.info()
print hr_data.describe()


#Convert some features to numeric
sales_unique = hr_data['sales'].unique().tolist()
sales_encoder = preprocessing.LabelEncoder()
sales_encoder.fit(sales_unique)

hr_data['sales'] = sales_encoder.transform(hr_data['sales'])

salary_unique = hr_data['salary'].unique().tolist()
salary_encoder = preprocessing.LabelEncoder()
salary_encoder.fit(salary_unique)

hr_data['salary'] = salary_encoder.transform(hr_data['salary'])


# Separate data in training and testing
X = hr_data.drop(['left'], axis=1)
y = hr_data['left']

# Normalization is required for NN
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns=X.columns)


# Set validation parameters
sample_size = int(0.7 * hr_data.shape[0])
sample_seed = 0

# Separate training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=sample_size, random_state=sample_seed)


# Build neural net
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=3)
act1 = mx.symbol.Activation(fc1, name='sig1', act_type="sigmoid")
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=2)
act2 = mx.symbol.Activation(fc2, name='sig2', act_type="sigmoid")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=2)
mlp = mx.symbol.SoftmaxOutput(fc3, name='softmax')

mx.viz.plot_network(mlp)


# Train the net
import logging
logging.basicConfig(level=logging.INFO)

mod = mx.mod.Module(symbol=mlp, context=mx.cpu(), data_names=['data'], label_names=['softmax_label'])

batch_size = 5
learning_rate = 0.3

train_iter = mx.io.NDArrayIter(data={'data':X_train.values},
                            label={'softmax_label':y_train.values},
                            batch_size=batch_size)

eval_iter = mx.io.NDArrayIter(data={'data':X_test.values},
                            label={'softmax_label':y_test.values},
                            batch_size=batch_size)

mod.fit(train_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':learning_rate},
        eval_metric='accuracy',
        num_epoch=10)


y_pred = mod.predict(eval_iter).asnumpy()
y_pred = map(lambda x: 1 if x[0] < 0.5 else 0, y_pred)

confusion_matrix(y_test, y_pred)

# Calculo métricas de evaluación
score = mod.score(eval_iter, ['mse', 'acc'])
score