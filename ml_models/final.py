import CNN as cnn_classifier
import LSTM as lstm_classifier
import csv
from bayes import run_tfIdf_classify,run_tfIdf_classify_reason,run_SLFeatures_classify,run_SLFeatures_classify_reason
from word2vec import getTestSet,getTestSet_Negative
file_path = 'Tweets.csv'
if __name__ == "__main__":
    list_tweet = []
    with open (file_path,"r",encoding='UTF-8') as f:
        reader = csv.reader(f)
        list_tweet = list(reader)
    
    list_tweet = list_tweet[1:]
    text_set,label_set = getTestSet()
    r_text_set,r_label_set = getTestSet_Negative()
    lstm_a=lstm_classifier.runAccurancy(text_set,label_set,"sen")
    lstm_a_r=lstm_classifier.runAccurancy(r_text_set,r_label_set,"reason")
    cnn_a = cnn_classifier.runAccurancy(text_set,label_set)
    cnn_a_r=cnn_classifier.runAccurancyReason(r_text_set,r_label_set)
    print("The accuracy of cnn to predict sentiment is :",cnn_a)
    print("The accuracy of cnn to negative reason is :",cnn_a_r)
    print("The accuracy of lstm to predict sentiment is :",lstm_a)
    print("The accuracy of lstm to negative reason is :",lstm_a_r)   
    bayes_tf_a=run_tfIdf_classify(list_tweet)
    print("The accuracy of bayes with tf-idf to predict sentiment is :",bayes_tf_a)
    bayes_tf_reason = run_tfIdf_classify_reason(list_tweet)
    print("The accuracy of bayes with tf-idf to predict negative reason is :",bayes_tf_reason)
    bayes_sl_a = run_SLFeatures_classify(list_tweet)
    print("The accuracy of bayes with SL to predict  sentiment is :",bayes_sl_a)
    bayes_sl_a = run_SLFeatures_classify_reason(list_tweet)
    print("The accuracy of bayes with SL to predict negative reason is :",bayes_sl_a)