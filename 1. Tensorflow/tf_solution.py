import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Different parameters for learning
learning_rate = 0.01
training_epochs = 1000
display_step = 100
np.random.seed(0)
x_name = 'sys'      # sys, dys or hr


df = pd.read_csv('mon.csv')

# Train test split
X = df[x_name]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0, stratify=y)

# Create placeholder for providing inputs
X = tf.placeholder("float")
y = tf.placeholder("float")

# create weights and bias and initialize with random number
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Construct a linear model using Y=WX+b
pred = tf.add(tf.multiply(X, W), b)

# Calculate Mean squared error
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*len(df.index))

# Gradient descent to minimize mean sequare error
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("Training started")

    # Fit training data
    for epoch in xrange(training_epochs):
        for(X_i, y_i) in zip(X_train, y_train):
            sess.run(optimizer, feed_dict={X:X_i, y:y_i})

        # Display epoch info after n steps
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X:X_train, y:y_train})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Training completed")
    training_cost = sess.run(cost, feed_dict={X:X_train, y:y_train})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Plotting doesn't make much sense with dichotomous y
    # # Plot line with fitted data
    # plt.plot(X_train, y_train, 'ro', label='Original data')
    # plt.plot(X_train, sess.run(W) * X_train + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()

    # Testing 
    print("Testing started")

    #Calculate Mean square error
    print("Calculate Mean square error")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * X_test.shape[0]),
        feed_dict={X: X_test, y: y_test})  # same function as cost above

    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    # # Plotting doesn't make much sense again
    # plt.plot(X_test, y_test, 'bo', label='Testing data')
    # plt.plot(X_train, sess.run(W) * X_train + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()