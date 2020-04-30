import tensorflow as tf
import word2vec
import numpy as np
from Cnn_Model import CNN
import word2vec
import datetime
from LSTM import LSTM
wordVectors = np.load('wordVectors.npy')
batchSize = 25
numClasses = 3
maxSeqLength = 20
SENTIMENT_INDEX=1
REASON_INDEX=3
AIRLINE_INDEX=5
TEXT_INDEX=10
iterations = 5000
embedding_size = 300


def runReason():
    tf.reset_default_graph()
    board_dir = './tensorboard/text_cnn_reason'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    model = CNN(len(word2vec.negative_reason))
    saver = tf.train.Saver()
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(board_dir)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print("start train")
        for i in range(iterations):
            train_set,label = word2vec.getTrainSet_Negative()
            feed_dict = model.feed_data(train_set,label,0.75)
            _,train_summary,train_loss,train_accuracy = session.run([model.train_step,merged_summary,model.loss,model.accuracy],feed_dict=feed_dict)
            if i % 100 ==0 and  i!=0:
                saver.save(session,"models/cnn_reson/pretrained.ckpt",global_step=i)
                print("saving model")
                print("iteration {}/{}".format(i+1,iterations))
                print("train loss is {}".format(train_loss))
                print("train_accuracy is {}".format(train_accuracy)) 
                writer.add_summary(train_summary,i)
        writer.close()
def runSen():
    tf.reset_default_graph()
    board_dir = './tensorboard/text_cnn'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    model = CNN(3)
    saver = tf.train.Saver()
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(board_dir)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print("start train")
        for i in range(iterations):
            train_set,label = word2vec.getTrainSet()
            feed_dict = model.feed_data(train_set,label,0.75)
            _,train_summary,train_loss,train_accuracy = session.run([model.train_step,merged_summary,model.loss,model.accuracy],feed_dict=feed_dict)
            if i % 100 ==0 and  i!=0:
                saver.save(session,"models/cnn/pretrained.ckpt",global_step=i)
                print("saving model")
                print("iteration {}/{}".format(i+1,iterations))
                print("train loss is {}".format(train_loss))
                print("train_accuracy is {}".format(train_accuracy))
                writer.add_summary(train_summary,i)
        writer.close()
def predictSen(tweet):
    tf.reset_default_graph()
    mode = CNN(3)
    with tf.Session() as sess:
        saver=saver = tf.train.Saver()
        ckp = tf.train.get_checkpoint_state('./models/cnn')
        saver.restore(sess,ckp.model_checkpoint_path)
        print("load "+ ckp.model_checkpoint_path)
        vector = [word2vec.tweet2vec(tweet) for i in range(25)]
        res = sess.run(mode.prediction,{mode.input_data:vector,mode.keep_prob:1.0})
        print(res)
        return res[0]

def predictReason(tweet):
    tf.reset_default_graph()
    mode = CNN(len(word2vec.negative_reason))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckp = tf.train.get_checkpoint_state('./models/cnn_reson')
        saver.restore(sess,ckp.model_checkpoint_path)
        vector = [word2vec.tweet2vec(tweet) for i in range(25)]
        res = sess.run(mode.prediction,{mode.input_data:vector,mode.keep_prob:1.0})
        return res[0]

def runAccurancy(x,y):
    tf.reset_default_graph()
    model = CNN(3)
    with tf.Session() as session:
        saver = tf.train.Saver()
        ckp = tf.train.get_checkpoint_state('./models/cnn')
        saver.restore(session,ckp.model_checkpoint_path)
        feed_dict = model.feed_data(x,y,1)
        train_accuracy = session.run([model.accuracy],feed_dict=feed_dict)
        return train_accuracy

def runAccurancyReason(x,y):
    tf.reset_default_graph()
    model = CNN(len(word2vec.negative_reason))
    with tf.Session() as session:
        saver = tf.train.Saver()
        ckp = tf.train.get_checkpoint_state('./models/cnn_reson')
        saver.restore(session,ckp.model_checkpoint_path)
        feed_dict = model.feed_data(x,y,1)
        accuracy = session.run([model.accuracy],feed_dict=feed_dict)
        return accuracy
if __name__ == "__main__":
    #runSen()
    #runReason()

    list = word2vec.getNegiaviveWithReason()
    classes = word2vec.negative_reason
    for i in range(30):
        print("origin"+list[i][REASON_INDEX+1])
        print(classes[predictReason(list[i][TEXT_INDEX+1])])
        print("------")
