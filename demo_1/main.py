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
