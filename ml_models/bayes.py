#This is bayes classifer
import nltk
import csv
import re
import word2vec
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.naive_bayes import MultinomialNB
file_path = 'Tweets.csv'
tweet_set = []
SENTIMENT_INDEX=1
REASON_INDEX=3
AIRLINE_INDEX=5
TEXT_INDEX=10
stoplist = nltk.corpus.stopwords.words("english")
SL_path = "subjclueslen1-HLTEMNLP05.tff"
label =1 
#process raw tweet,delete @name
def tweet_process(tweet):
    #remove @name
    new_tweet = re.sub(r'@\S+',"",tweet)
    document_words = nltk.word_tokenize(new_tweet)
    document_words = [word.lower() for word in document_words if (word not in stoplist) and (len(word)>2) and (re.search('[A-Za-z]+',word) != None)]
    return document_words
#load SL 
def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict
#extract positive negative features
def SL_features(document,SL):
    document_words = set(document)
    features = {}
    weakPos = 0
    weakNeg =0
    strongPos = 0
    strongNeg = 0
    features['positivecount'] =0
    features['negativecount'] = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
    return features  
def getTf_Idf(word,sent,documents):
    corpus = nltk.text.TextCollection(documents)
    idf = corpus.idf(word)
    tf = corpus.tf(word,sent)
    return idf*tf

def tfIdf_features(sentence,documents):
    words = set(sentence)
    features = {}
    for word in words:
        features['V_{}'.format(word)] = getTf_Idf(word,sentence,documents)
    return features
#tf-idf bayes classifier
def run_tfIdf_classify(list):
    process_tweets = [" ".join(tweet_process(tweet[TEXT_INDEX])) for tweet in list]
    tag_list = [tweet[SENTIMENT_INDEX] for tweet in list]
    vectorizer = CountVectorizer()  
    X = vectorizer.fit_transform(process_tweets)
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(X)
    tfArray =  tfidf.toarray()
    clm = MultinomialNB().fit(tfArray[0:10000],tag_list[0:10000])
    return clm.score(tfArray[10001:],tag_list[10001:])

#tf-idf negative reason
def run_tfIdf_classify_reason(list):
    process_tweets = [" ".join(tweet_process(tweet[TEXT_INDEX])) for tweet in list if len(tweet[REASON_INDEX])>0]
    tag_list = [tweet[REASON_INDEX] for tweet in list]
    vectorizer = CountVectorizer()  
    X = vectorizer.fit_transform(process_tweets)
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(X)
    tfArray =  tfidf.toarray()
    clm = MultinomialNB().fit(tfArray[0:7000],tag_list[0:7000])
    return clm.score(tfArray[9000:9025],tag_list[9000:9025])

def run_SLFeatures_classify(list):
    SL = readSubjectivity(SL_path)
    SL_features_list =[(SL_features(tweet_process(tweet[TEXT_INDEX]),SL),tweet[SENTIMENT_INDEX]) for tweet in list]
    train_set, test_set = SL_features_list[0:10000], SL_features_list[10001:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    acc = nltk.classify.accuracy(classifier, test_set)
    return acc
def run_SLFeatures_classify_reason(list):
    SL = readSubjectivity(SL_path)
    SL_features_list =[(SL_features(tweet_process(tweet[TEXT_INDEX]),SL),tweet[REASON_INDEX]) for tweet in list if len(tweet[REASON_INDEX])>0]
    train_set, test_set = SL_features_list[0:7000], SL_features_list[9000:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    acc = nltk.classify.accuracy(classifier, test_set)
    return acc

if __name__ == "__main__":
    
    with open (file_path,"r",encoding='UTF-8') as f:
        reader = csv.reader(f)
        list_tweet = list(reader)
    SL = readSubjectivity(SL_path)
    SL_features_list =[(SL_features(tweet_process(tweet[TEXT_INDEX]),SL),tweet[SENTIMENT_INDEX]) for tweet in list_tweet[1:]]
    #seniment classify
    train_set, test_set = SL_features_list[4000:], SL_features_list[:4000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    acc = nltk.classify.accuracy(classifier, test_set)
    print(acc)




    
