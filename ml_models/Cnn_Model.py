#class model for CNN
import tensorflow as tf
import numpy as np
wordVectors = np.load('wordVectors.npy')
batchSize = 25
numClasses = 3
maxSeqLength = 20
SENTIMENT_INDEX=1
REASON_INDEX=3
AIRLINE_INDEX=5
TEXT_INDEX=10
embedding_size = 50

class CNN():
    def __init__(self,numClasses):
        self.input_data = tf.placeholder(tf.int32,[None,maxSeqLength])
        self.labels  = tf.placeholder(tf.float32,[None,numClasses])
        self.embedding = tf.nn.embedding_lookup(wordVectors,self.input_data)
        self.embedding= tf.expand_dims(self.embedding,-1)
        self.total_pool = []
        self.keep_prob = tf.placeholder(tf.float32)
        self.global_step =tf.Variable(0, trainable=False, name='global_step')
        ##TEXT的cnn的卷积不是层层递进，而是并列选出几个features再合在一起
        self.w_conv1 = self.weight_variable([3,embedding_size,1,32])
        self.b_conv1 = self.bias_variable([32])
        self.conv1 = tf.nn.relu(self.conv2d(self.embedding,self.w_conv1)+self.b_conv1)
        self.pool1 = self.max_pool(self.conv1,3)
        self.total_pool.append(self.pool1)
        self.w_conv2 = self.weight_variable([4,embedding_size,1,32])
        self.b_conv2 = self.bias_variable([32])
        self.conv2 = tf.nn.relu(self.conv2d(self.embedding,self.w_conv2)+self.b_conv2)
        self.pool2 = self.max_pool(self.conv2,4)
        self.total_pool.append(self.pool2)
        self.w_conv3 = self.weight_variable([5,embedding_size,1,32])
        self.b_conv3 = self.bias_variable([32])
        self.conv3 =tf.nn.relu(self.conv2d(self.embedding,self.w_conv3)+self.b_conv3)
        self.pool3 = self.max_pool(self.conv3,5)
        self.total_pool.append(self.pool3)
        total_filter = 32*3
        self.pool = tf.concat(self.total_pool,3)
        self.pool_flat = tf.reshape(self.pool,[-1,total_filter])
        self.fc_drop = tf.nn.dropout(self.pool_flat,self.keep_prob)
        self.w_fc = self.weight_variable([total_filter,numClasses])
        self.b_fc = self.bias_variable([numClasses])
        self.score = tf.matmul(self.fc_drop,self.w_fc)+self.b_fc
        self.y_conv = tf.nn.softmax(self.score)
        self.prediction = tf.argmax(self.y_conv,1)

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score,labels=self.labels)
        self.loss = tf.reduce_mean(losses)

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss,global_step=self.global_step)
        self.saver = tf.train.Saver()

        correction_prediction = tf.equal(self.prediction,tf.argmax(self.labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(correction_prediction, 'float32'), name='accuracy')
    @staticmethod
    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    @staticmethod
    def bias_variable(shape):
        return tf.Variable(tf.constant(0.1,shape=shape))
    @staticmethod
    def conv2d(x,weight):
        return tf.nn.conv2d(x,weight,strides=[1,1,1,1],padding='VALID')
    @staticmethod
    def max_pool(x,filter):
        return tf.nn.max_pool(x,ksize=[1,maxSeqLength-filter+1,1,1],strides=[1,1,1,1],padding='VALID')

    def feed_data(self,input,labels,keep_prob):
        feed_dict = {
            self.input_data:input,
            self.labels:labels,
            self.keep_prob:keep_prob
        }
        return feed_dict
    def evaluate(self,sess,x,y):
        feed_dict = self.feed_data(x, y, 1.0)
        loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return loss,accuracy
