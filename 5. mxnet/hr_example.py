import mxnet as mx
import pandas as pd
from sklearn.model_selection import train_test_split

# Read data and basic statistics
hr_data = pd.read_csv("data/hr.csv")
hr_data.head()
hr_data.info()
hr_data.describe()

# Convert some variables to factors
hr_data[['Work_accident', 'promotion_last_5years']] = \
    hr_data[['Work_accident', 'promotion_last_5years']].astype(object)

sample_size = int(0.7 * hr_data.shape[0])
sample_seed = 0

X = hr_data.drop('left', axis=1)
y = hr_data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=sample_size, random_state=sample_seed)

net = mx.symbol.Variable('data')
net = mx.symbol.FullyConnected(net, name='fc1', num_hidden=3)
net = mx.symbol.Activation(net, name='sig1', act_type="sigmoid")
net = mx.symbol.FullyConnected(net, name='fc2', num_hidden=2)
net = mx.symbol.Activation(net, name='sig2', act_type="sigmoid")
mlp = mx.symbol.SoftmaxOutput(net, name='softmax', multi_output=True)

nd_iter = mx.io.NDArrayIter(data={'data':X_train},
                            label={'softmax_label':y_train},
                            batch_size=batch_size)
print(nd_iter.provide_data)
print(nd_iter.provide_label)



mlpmodel <- mx.mlp(data = train.preds
                   ,label = train.target
                   ,hidden_node = c(3,2) #two layers- 1st Layer with 3 nodes and 2nd with 2 nodes
                   ,out_node = 2 #Number of output nodes
                   ,activation="sigmoid" #activation function for hidden layers
                   ,out_activation = "softmax")

                   ,num.round = 10 #number of iterations
                   ,array.batch.size = 5 #Batch size for updating weights
                   ,learning.rate = 0.03 #same as step size
                   ,eval.metric= mx.metric.accuracy
                   ,eval.data = list(data = test.preds, label = test.target))

