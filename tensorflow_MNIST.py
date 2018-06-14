
import tensorflow as tf
#from keras.datasets import cifar10
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100


# Network Parameters
#n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
n_samples = mnist.train.num_examples


_IMAGE_SIZE = 28
_IMAGE_CHANNELS = 1
_NUM_CLASSES = 10

x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')    
      
y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

conv = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
       
conv = tf.layers.conv2d(
            inputs=conv,
            filters=64,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
drop = tf.layers.dropout( inputs=pool,rate=0.5)
flat = tf.layers.flatten(drop,name=None)

#artificial neural network starts

fc = tf.layers.dense(inputs=flat, units=1500, activation=tf.nn.relu)
drop = tf.layers.dropout(fc, rate=0.6)
softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, activation=tf.nn.softmax)
pred_ = tf.argmax(softmax, axis=1)

y.shape

# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

loss.shape
y.shape

# Initializing the variables
init = tf.initialize_all_variables()


# Launch the session
sess = tf.InteractiveSession()

# Intialize all the variables
sess.run(init)


# Training Epochs
# Essentially the max amount of loops possible before we stop
# May stop earlier if cost/loss limit was set
for epoch in range(training_epochs):

    # Start with cost = 0.0
    avg_cost = 0.0

    # Convert total number of batches to integer
    total_batch = int(n_samples/batch_size)

    # Loop over all batches
    for i in range(total_batch):

        # Grab the next batch of training data and labels
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Feed dictionary for optimization and loss value
        # Returns a tuple, but we only need 'c' the cost
        # So we set an underscore as a "throwaway"
        _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

        # Compute average loss
        avg_cost += c / total_batch

    print("Epoch: {} cost={:.4f}".format(epoch+1,avg_cost))

print("Model has completed {} Epochs of Training".format(training_epochs))


#ITS ADVISABLE TO RUN TRAINING PART OF THE CODE FIRST AND THEN UNCOMMENT THE LOWER TEST PART AND THE REST.
"""

# Test model
correct_predictions = tf.equal(tf.argmax(pred_, 1), tf.argmax(y, 1))

print(correct_predictions[0])

correct_predictions = tf.cast(correct_predictions, "float")

print(correct_predictions[0])


accuracy = tf.reduce_mean(correct_predictions)
type(accuracy)

mnist.test.labels


print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))









"""








































































































