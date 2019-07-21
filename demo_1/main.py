import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import the MNIST data set and store the image data
# Using one_hot encoding as vector of binary values to represent categorical values 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MSINT_data/",one_hot=True)

# using mnist to find size of dataset we imported
mn_train = mnist.train.num_examples     #55,000
mn_validation = mnist.validation.num_examples       #5000
n_test = mnist.test.num_examples        #10,000

# Defining model architecture
mn_input = 784
mn_hidden_1 = 512
mn_hidden_2 = 256
mn_hidden_3 = 128
mn_hidden_4 = 64
mn_hidden_5 = 32
mn_output = 10

# defining the hyperparemeters
dropout = 0.5       #threshold at which we eliminate some units at random, prevents overfitting
batch_size = 128        #training examples used at each step
mn_iterations = 1000        #how many times we go through training steps1
learning_rate = 1e-4    #how much parameters will adjust at each learning process

#defining tensors as placeholders
X = tf.placeholder("float", [None, mn_input])
Y = tf.placeholder("float", [None, mn_output])
keep_prob = tf.placeholder(tf.float32)

#parameters that network will be updating in the training process are weight and bias
#so we initialize these rather than using empty placeholders
weights = {
    'w1': tf.Variable(tf.truncated_normal([mn_input, mn_hidden_1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([mn_hidden_1, mn_hidden_2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([mn_hidden_2, mn_hidden_3], stddev=0.1)),
    'w4': tf.Variable(tf.truncated_normal([mn_hidden_3, mn_hidden_4], stddev=0.1)),
    'w5': tf.Variable(tf.truncated_normal([mn_hidden_4, mn_hidden_5], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([mn_hidden_5, mn_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[mn_hidden_1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[mn_hidden_2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[mn_hidden_3])),
    'b4': tf.Variable(tf.constant(0.1, shape=[mn_hidden_4])),
    'b5': tf.Variable(tf.constant(0.1, shape=[mn_hidden_5])),
    'out': tf.Variable(tf.constant(0.1, shape=[mn_output]))
}

# setup layers that will manipulate tensors
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_4 = tf.add(tf.matmul(layer_3, weights['w4']), biases['b4'])
layer_5 = tf.add(tf.matmul(layer_4, weights['w5']), biases['b5'])
layer_drop = tf.nn.dropout(layer_5, keep_prob)
output_layer = tf.matmul(layer_5, weights['out']) + biases['out']