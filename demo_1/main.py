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