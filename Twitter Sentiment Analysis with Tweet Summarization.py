# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:59:22 2018

@author: Rahul Chanderiya
"""

# Importing the libraries
import tweepy
import re
import pickle
import matplotlib.pyplot as plt
from tweepy import OAuthHandler
import nltk
import heapq
import numpy as np




# Text Summarization

def summarize(text):

    # Preprocessing the data
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'(\@|\#)(\S)+(\b)',' ',text)
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', text)
    clean_text = text.lower()
    clean_text = re.sub(r'\W',' ',clean_text)
    clean_text = re.sub(r'\d',' ',clean_text)
    clean_text = re.sub(r'(\s)+',' ',clean_text)
    
    # Tokenize sentences
    sentences = nltk.sent_tokenize(text)
    
    # Stopword list
    stop_words = nltk.corpus.stopwords.words('english')
    
    # Word counts 
    word2count = {}
    for word in nltk.word_tokenize(clean_text):
        if word not in stop_words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1
    
    # Converting counts to weights
    for key in word2count.keys():
        word2count[key] = word2count[key]/max(word2count.values())
        
    # Product sentence scores    
    sent2score = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word2count.keys():
                if len(sentence.split(' ')) < 25:
                    if sentence not in sent2score.keys():
                        sent2score[sentence] = word2count[word]
                    else:
                        sent2score[sentence] += word2count[word]
                        
    # Gettings best 5 lines             
    best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)
    
    summary = []
    for sentence in best_sentences:
        #summary = summary + "\n" + sentence;
        summary.append(sentence)
    
    return summary




# Twitter Sentiment Analysis

def twitter_sentiment_analysis(topic, num_of_tweets):
    
    # Initializing the keys
    consumer_key = 'yoIwFkjZGYDa49aO16XqSNqcN'
    consumer_secret = 'gl4LQOItV7Z1aFwNrlvaiKJ3t8o8h99blMIAmnmdHxYjzjRAxO' 
    access_token = '624310916-E7fDF2IE8P6bfY1oVFglASf6F8RnxMd3vgSXFqnZ'
    access_secret ='ID9JcoXHsDcKtvNcnmBGcCQhUlO0wmwAxBJ6LCesiUAas'
    
    # Initializing the tokens
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    args = [topic];
    api = tweepy.API(auth,timeout=10)
    
    # Fetching the tweets
    list_tweets = []
    
    query = args[0]
    if len(args) == 1:
        for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent',geocode="22.1568,89.4332,500km").items(num_of_tweets):
            list_tweets.append(status.text)
            
    # Loading the vectorizer and classfier
    with open('classifier.pickle','rb') as f:
        classifier = pickle.load(f)
        
    with open('tfidfmodel.pickle','rb') as f:
        tfidf = pickle.load(f)    
        
    total_pos = 0
    total_neg = 0
    
    positive_text = ""
    negative_text = ""
    
    positive_tweets = []
    negative_tweets = []
    
    # Preprocessing the tweets and predicting sentiment
    for tweet in list_tweets:
        
        tweetx = tweet
        tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
        tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
        tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
        tweet = tweet.lower()
        tweet = re.sub(r"that's","that is",tweet)
        tweet = re.sub(r"there's","there is",tweet)
        tweet = re.sub(r"what's","what is",tweet)
        tweet = re.sub(r"where's","where is",tweet)
        tweet = re.sub(r"it's","it is",tweet)
        tweet = re.sub(r"who's","who is",tweet)
        tweet = re.sub(r"i'm","i am",tweet)
        tweet = re.sub(r"she's","she is",tweet)
        tweet = re.sub(r"he's","he is",tweet)
        tweet = re.sub(r"they're","they are",tweet)
        tweet = re.sub(r"who're","who are",tweet)
        tweet = re.sub(r"ain't","am not",tweet)
        tweet = re.sub(r"wouldn't","would not",tweet)
        tweet = re.sub(r"shouldn't","should not",tweet)
        tweet = re.sub(r"can't","can not",tweet)
        tweet = re.sub(r"couldn't","could not",tweet)
        tweet = re.sub(r"won't","will not",tweet)
        tweet = re.sub(r"\W"," ",tweet)
        tweet = re.sub(r"\d"," ",tweet)
        tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
        tweet = re.sub(r"\s+[a-z]$"," ",tweet)
        tweet = re.sub(r"^[a-z]\s+"," ",tweet)
        tweet = re.sub(r"\s+"," ",tweet)
        sent = classifier.predict(tfidf.transform([tweet]).toarray())
        
        if sent[0] == 1:
            total_pos += 1
            positive_tweets.append(tweetx);
            positive_text = positive_text + tweetx + "\n"
        else:
            total_neg += 1
            negative_tweets.append(tweetx);
            negative_text = negative_text + tweetx + "\n"
    
        
    # Visualizing the results
    objects = ['Positive','Negative']
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos,[total_pos,total_neg],alpha=0.5)
    plt.xticks(y_pos,objects)
    plt.ylabel('Number')
    plt.title('Number of Postive and Negative Tweets')
    
    plt.show()
    print("\n")
    
    
    # Showing Positive and Negative Tweets with respective Summary
    print("Positive Tweets : \n")
    for tweet in positive_tweets:
        print (tweet + "\n")
    print ("Summary : \n")
    print( summarize(positive_text) )
    
    print("\n\n")
    
    print("Negative Tweets : \n")
    for tweet in negative_tweets:
        print (tweet + "\n")
    print ("Summary : \n")
    print( summarize(negative_text) )

    return




# Main Function
def main():
  twitter_sentiment_analysis("modi", 100)
  

main()


