import numpy as np
import tensorflow as tf
import nltk
import re
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from pathlib import Path
from random import randint
negative_reason=['Bad Flight', "Can't Tell", 'Late Flight', 'Customer Service Issue', 'Flight Booking Problems', 'Lost Luggage', 'Flight Attendant Complaints', 'Cancelled Flight', 'Damaged Luggage', 'longlines']
sens = ['positive','negative','neutral']
file_path = 'Tweets.csv'
tweet_set = []
SENTIMENT_INDEX=1
REASON_INDEX=3
AIRLINE_INDEX=5
TEXT_INDEX=10
REASON_INDEX=3
stoplist = nltk.corpus.stopwords.words("english")
wordsList = np.load('wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')
list_tweet = []
numWords=[]
maxSeqLength=20
batchSize=25

def tweet2vec(tweet):
    clean_tweet = tweet_process(tweet)
    indexc = 0
    vec = np.zeros((maxSeqLength),dtype='int32')
    for word in clean_tweet:
        try:
            vec[indexc] = wordsList.index(word)
        except ValueError:
            vec[indexc] = 399999
        indexc = indexc+1
        if indexc >= maxSeqLength:
            break
    return vec

def tweet_process(tweet):
    #remove @name
    new_tweet = re.sub(r'@\S+',"",tweet)
    document_words = nltk.word_tokenize(new_tweet)
    document_words = [word.lower() for word in document_words if (word not in stoplist)  and (re.search('[A-Za-z]+',word) != None)]
    return document_words
def transfor2ids(text_list,index,out_put):
    numFiles = len(text_list)
    ids = np.zeros((numFiles,maxSeqLength),dtype='int32')
    for i in range(numFiles):
        words = text_list[i][index]
        indexCounter=0
        for word in words:
            try:
                ids[i][indexCounter]=wordsList.index(word)
            except ValueError:
                ids[i][indexCounter]=399999 
            indexCounter = indexCounter+1
            if indexCounter >= maxSeqLength:
                break
    np.save(out_put,ids)
def getTrainSet():
    with open (file_path,"r",encoding='UTF-8') as f:
        reader = csv.reader(f)
        list_tweet = list(reader)
    ids = np.load('idsMatrix.npy')
    list_tweet.pop(0)
    labels = []
    arr = np.zeros([batchSize,maxSeqLength])
    for i in range(batchSize):
        index = randint(0,10000)
        if list_tweet[index][SENTIMENT_INDEX]=='positive':
            labels.append([1,0,0])
        elif list_tweet[index][SENTIMENT_INDEX]=='negative':
            labels.append([0,1,0])
        else:
            labels.append([0,0,1])
        arr[i] = ids[index]
    return arr,labels
def getTestSet():
    with open (file_path,"r",encoding='UTF-8') as f:
        reader = csv.reader(f)
        list_tweet = list(reader)
    ids = np.load('idsMatrix.npy')
    list_tweet.pop(0)
    labels = []
    arr = np.zeros([1000,maxSeqLength])
    for i in range(1000):
        num = len(list_tweet)-1-i
        if list_tweet[num][SENTIMENT_INDEX]=='positive':
            labels.append([1,0,0])
        elif list_tweet[num][SENTIMENT_INDEX]=='negative':
            labels.append([0,1,0])
        else:
            labels.append([0,0,1])
        arr[i] = ids[num]
    return arr,labels

def getTrainSet_Negative():  
    list_tweet = getNegiaviveWithReason()
    ids = np.load('ids_negative.npy')
    labels = []
    arr = np.zeros([batchSize,maxSeqLength])
    for i in range(batchSize):
        index = randint(0,7000)
        label = [0]*len(negative_reason)
        try:
            re_indx = negative_reason.index(list_tweet[index][REASON_INDEX+1])
        except ValueError:
            re_indx = 1
        label[re_indx] = 1
        labels.append(label)
        arr[i] = ids[index]
    return arr,labels

def getTestSet_Negative():
    list_tweet = getNegiaviveWithReason()
    ids = np.load('ids_negative.npy')
    labels = []
    arr = np.zeros([1000,maxSeqLength])
    for i in range(1000):
        index = len(list_tweet)-1-i
        label = [0]*len(negative_reason)
        try:
            re_indx = negative_reason.index(list_tweet[index][REASON_INDEX+1])
        except ValueError:
            re_indx = 1
        label[re_indx] = 1
        labels.append(label)
        arr[i] = ids[index]
    return arr,labels   

def getNegiaviveWithReason():
    negative_path = "negative.csv"
    my_file = Path(negative_path)
    if my_file.exists():
        with open(negative_path,"r",encoding='UTF-8') as f:
            reader = csv.reader(f)
            negative_list = list(reader)
            negative_list.pop(0)
            return negative_list
    negative_tweet = []
    with open (file_path,"r",encoding='UTF-8') as f:
        reader = csv.reader(f)
        negative_tweet = list(reader)
    title = negative_tweet.pop(0)
    negative_tweet = [tweet for tweet in negative_tweet if len(tweet[REASON_INDEX])>0 and tweet[SENTIMENT_INDEX]=='negative']
    negatives = pd.DataFrame(columns=title,data=negative_tweet)
    negatives.to_csv("./negative.csv",encoding='utf-8')
    return negative_tweet



if __name__ == "__main__":
    '''
    with open (file_path,"r",encoding='UTF-8') as f:
        reader = csv.reader(f)
        list_tweet = list(reader)
    list_tweet.pop(0)
    for tweet in list_tweet:
        clean_tweet = tweet_process(tweet[TEXT_INDEX])
        tweet[TEXT_INDEX] = clean_tweet
        numWords.append(len(clean_tweet))
    numFiles = len(numWords)
    transfor2ids(list_tweet,TEXTX_INDEX,"idsMatrix")
    '''
    '''
    negative_list = getNegiaviveWithReason()
    for tweet in negative_list:
        clean_tweet = tweet_process(tweet[TEXT_INDEX+1])
        tweet[TEXT_INDEX+1] = clean_tweet
    #transfor2ids(negative_list,TEXT_INDEX+1,"ids_negative")
    '''
    '''
    reasons = []
    for tweet in negative_list:
        try:
           ni = reasons.index(tweet[REASON_INDEX+1])
        except ValueError:
            reasons.append(tweet[REASON_INDEX+1])
            print(tweet[REASON_INDEX+1])
    print(reasons)
    '''
    print(wordVectors[1])