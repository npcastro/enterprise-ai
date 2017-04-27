import numpy
from keras.models import Sequential
from keras.layers import Dense


# Fix the random seed for reproducibility
numpy.random.seed(7)

# Load the dataset
data = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# Separate between training and testing data
X = data[:, 0:8]
y = data[:, 8]


# Create the neural network
model = Sequential()

# Add a layer -- number neuros, inputs, and activation function
# The dense layer is a fully connected layer
model.add(Dense(12, input_shape=X[0].shape, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Model compilation
# Requires loss function, optimizer function and additional metrics (optional)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model fitting
# epochs: number of iterations the process will run through the dataset
# batch_size: number of instances evaluated before the model updates the weights
model.fit(X, y, epochs=150, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, y)
print("\n{}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))