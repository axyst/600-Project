import numpy as np
import tensorflow as tf
from random import randint
import re
import csv
import nltk
import datetime
import word2vec
maxSeqLength = 20
batchSize = 25
lstmUnits = 64
numClasses = 3
iterations = 5000
SENTIMENT_INDEX=1
REASON_INDEX=3
AIRLINE_INDEX=5
TEXT_INDEX=10
wordVectors = np.load('wordVectors.npy')
class LSTM():
    def __init__(self,numClasses):
        self.labels = tf.placeholder(tf.float32,[None,numClasses])
        self.input_data = tf.placeholder(tf.int32,[None,maxSeqLength])
        self.data = tf.nn.embedding_lookup(wordVectors,self.input_data)
        self.lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        self.lstmCell = tf.contrib.rnn.DropoutWrapper(cell=self.lstmCell, output_keep_prob=0.75)
        self.value, _ = tf.nn.dynamic_rnn(self.lstmCell, self.data, dtype=tf.float32)
        self.weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        self.bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        self.value = tf.transpose(self.value, [1, 0, 2])
        self.last = tf.gather(self.value, int(self.value.get_shape()[0]) - 1)
        self.prediction = (tf.matmul(self.last, self.weight) + self.bias)
        self.result = tf.arg_max(self.prediction,1)
        self.correctPred = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def feed_data(self,input,labels):
        feed_dict = {
            self.input_data:input,
            self.labels:labels
        }
        return feed_dict
def runLstm(numClasses,out_put,getSet):
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32,[batchSize,numClasses])
    input_data = tf.placeholder(tf.int32,[batchSize,maxSeqLength])
    data = tf.Variable(tf.zeros([batchSize,maxSeqLength,300]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors,input_data)
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    with tf.InteractiveSession() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        print("start train")
        for i in range(iterations):
            nextBatch,nextBatchLabels =getSet()
            sess.run(optimizer,{input_data:nextBatch,labels:nextBatchLabels})
            if (i%50==0):
                summary = sess.run(merged,{input_data:nextBatch,labels:nextBatchLabels})
                writer.add_summary(summary,i)
            if(i%100==0 and i != 0):
                save_path =saver.save(sess,"models/out_put/pretrained.ckpt",global_step=i)
                print("saved sucess")
                loss_result = sess.run(loss,{input_data:nextBatch,labels:nextBatchLabels})
                accuracy_result = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
                print("iteration {}/{}".format(i+1,iterations))
                print("loss is {}".format(loss_result))
                print("accuracy is {}".format(accuracy_result))
        writer.close() 

def runSentiment():
    tf.reset_default_graph()
    model = LSTM(3)
    tf.summary.scalar('Loss', model.loss)
    tf.summary.scalar('Accuracy', model.accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/lstm_sentiment" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("start train")
        for i in range(iterations):
            nextBatch,nextBatchLabels =word2vec.getTrainSet()
            sess.run(model.optimizer,{model.input_data:nextBatch,model.labels:nextBatchLabels})
            if (i%50==0):
                summary = sess.run(merged,{model.input_data:nextBatch,model.labels:nextBatchLabels})
                writer.add_summary(summary,i)
            if(i%100==0 and i != 0):
                save_path =saver.save(sess,"models/lstm_sen/pretrained.ckpt",global_step=i)
                print("saved sucess")
                loss_result = sess.run(model.loss,{model.input_data:nextBatch,model.labels:nextBatchLabels})
                accuracy_result = sess.run(model.accuracy, {model.input_data: nextBatch, model.labels: nextBatchLabels})
                print("iteration {}/{}".format(i+1,iterations))
                print("loss is {}".format(loss_result))
                print("accuracy is {}".format(accuracy_result))
        writer.close() 

def runReason():
    tf.reset_default_graph()
    model = LSTM(len(word2vec.negative_reason))
    tf.summary.scalar('Loss', model.loss)
    tf.summary.scalar('Accuracy', model.accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/lstm_reason" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    sess = tf.InteractiveSession()
    writer = tf.summary.FileWriter(logdir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print("start train")
    for i in range(iterations):
        nextBatch,nextBatchLabels =word2vec.getTrainSet_Negative()
        feed_dict = model.feed_data(nextBatch,nextBatchLabels)
        sess.run(model.optimizer,feed_dict=feed_dict)
        if (i%50==0):
            summary = sess.run(merged,{model.input_data:nextBatch,model.labels:nextBatchLabels})
            writer.add_summary(summary,i)
        if(i%100==0 and i != 0):
            save_path =saver.save(sess,"models/lstm_reason/pretrained.ckpt",global_step=i)
            print("saved sucess")
            loss_result = sess.run(model.loss,feed_dict=feed_dict)
            accuracy_result = sess.run(model.accuracy,feed_dict=feed_dict)
            print("iteration {}/{}".format(i+1,iterations))
            print("loss is {}".format(loss_result))
            print("accuracy is {}".format(accuracy_result))
    writer.close() 

def predictReason(tweet):
    tf.reset_default_graph()
    model = LSTM(len(word2vec.negative_reason))
    with tf.Session() as sess:
        saver=saver = tf.train.Saver()
        ckp = tf.train.get_checkpoint_state('./models/lstm_reason')
        saver.restore(sess,ckp.model_checkpoint_path)
        print("load "+ ckp.model_checkpoint_path)
        vector = [word2vec.tweet2vec(tweet) for i in range(25)]
        res = sess.run(model.result,{model.input_data:vector})
        return res[0]
def predictSen(tweet):
    tf.reset_default_graph()
    model = LSTM(3)
    with tf.Session() as sess:
        saver=saver = tf.train.Saver()
        ckp = tf.train.get_checkpoint_state('./models/lstm_sen')
        saver.restore(sess,ckp.model_checkpoint_path)
        print("load "+ ckp.model_checkpoint_path)
        vector = [word2vec.tweet2vec(tweet) for i in range(25)]
        res = sess.run(model.result,{model.input_data:vector})
        return res[0]

def runAccurancy(x,y,type):
    tf.reset_default_graph()
    if type == "reason":
        model = LSTM(len(word2vec.negative_reason))
        model_path = './models/lstm_reason'
    else:
        model = LSTM(3)
        model_path = './models/lstm_sen'
    with tf.Session() as session:
        saver = tf.train.Saver()
        ckp = tf.train.get_checkpoint_state(model_path)
        saver.restore(session,ckp.model_checkpoint_path)
        _result = session.run(model.accuracy, {model.input_data: x, model.labels: y})
    return _result


if __name__ == "__main__":
    #sem
    #runSentiment()
    #reason
    runReason()
    '''
    list = word2vec.getNegiaviveWithReason()
    classes = word2vec.sens
    for i in range(30):
        print("origin"+list[i][REASON_INDEX+1])
        print(classes[predictSen(list[i][TEXT_INDEX+1])])
        print("------")
        '''