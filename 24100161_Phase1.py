#!/usr/bin/env python
# coding: utf-8

# In[74]:


import snscrape.modules.twitter as sntwitter
from sklearn.model_selection import train_test_split
import pandas as pd
import re


# # Machine Learning Project Phase 1
# ***
# ### Submitted by: 
# Saad Sher Alam (24100161)
# 
# ### Readme: 
# - Scraping Libarary: [snscrape](https://github.com/JustAnotherArchivist/snscrape)

# ## Part 1

# In[4]:


# Scraping tweets 

# list of tweets
tweet_data = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:SpursOfficial').get_items()):
    if i>999:
        break
    tweet_data.append([tweet.rawContent])


# In[7]:


from csv import writer 

# Adding column name 'tweets' to csv 
with open('SpursOfficial_part1.csv', 'a') as file: 
    wr = writer(file)
    wr.writerows([['tweets']]) 
    file.close()

# Writing all tweets to csv 
with open('SpursOfficial_part1.csv', 'a') as file: 
    wr = writer(file)
    wr.writerows(tweet_data)
    file.close() 


# In[96]:


# Checking csv
df = pd.read_csv("./SpursOfficial_part1.csv")
df.head(10)


# ### Part 2

# <b>Cleaning:</b> <br>
# - Convert tweets to lowercase 
# - Remove punctuation 
# - Remove numbers 
# - Remove single characters
# - Remove \n
# - Remove stop words (from Assignment 2)
# - Remove URLs

# In[97]:


# Getting stop words 
stop_words = {} 
f = open('./stop_words.txt', 'r')
for x in f:
    stop_words[(x.split("\n")[0])] = 1


# In[99]:


def remove_stop_words(review):
    cleaned_review = []
    words = review.split(" ")
    for word in words: 
        if word not in stop_words.keys(): 
            cleaned_review.append(word)
    return cleaned_review

def clean_tweets(dataset): 
    cleaned_tweets = [] 
    tweets = dataset.values
    for tweet in tweets: 
        tweet = tweet[0]
        # Converting to lowercase 
        clean = tweet.lower() 

        # Remove punctuation, numbers, and single characters 
        clean = re.sub(r'[^\w\s\@\#]', '', clean)
        clean = re.sub(r'\d', '', clean)
        clean = re.sub(r'\b[A-Z]\b', '', clean)

        # Removing /n 
        clean = clean.replace("\n", " ")

        # Removing stop words
        word_list = remove_stop_words(clean) 
        clean = " ".join(word_list)

        # Remove URLs (from assignment 0)
        clean = re.sub(r"https\S+", "", clean)
        
        cleaned_tweets.append([clean])
    return cleaned_tweets

cleaned_tweets = clean_tweets(df)


# In[100]:


# Adding column name 'tweets' to csv 
with open('SpursOfficial_part2.csv', 'a') as file: 
    wr = writer(file)
    wr.writerows([['tweets']]) 
    file.close()

# Writing all cleaned tweets to csv 
with open('SpursOfficial_part2.csv', 'a') as file: 
    wr = writer(file)
    wr.writerows(cleaned_tweets)
    file.close() 


# In[102]:


# Loading cleaned df
cleaned_df = pd.read_csv('SpursOfficial_part2.csv')
cleaned_df.head(10)


# ### Part 3
# 

# In[103]:


# Splitting Data 
train, test = train_test_split(cleaned_df, test_size=0.2, random_state=42)
train.head()


# In[108]:


def construct_vocabulary(dataframe):
    vocabulary = [] 
    all_tweets = dataframe.values
    for tweet in all_tweets: 
        tweet = str(tweet[0])
        tweet_words = tweet.split()
        for tweet_word in tweet_words: 
            if tweet_word not in vocabulary: 
                vocabulary.append(tweet_word)
    return vocabulary

def get_bag_of_words(tweet, vocabulary): 
    bag_of_words = {} 
    for word in vocabulary: 
        bag_of_words[word] = 0 
    
    tweet_words = tweet.split() 
    for word in tweet_words: 
        if word in bag_of_words.keys(): 
            bag_of_words[word] += 1 

    # Laplace smoothing  
    # Add 1 to count of training vocabulary 
    # alpha = 1
    for key in bag_of_words.keys(): 
        bag_of_words[key] += 1

    return bag_of_words


# In[109]:


vocab  = construct_vocabulary(train)
print("Ambient Dimensionality:",  len(vocab))
print(vocab)


# In[110]:


# Bag of words for training data 
train_tweets = train.values
train_bag_of_words = [] 
for tweet in train_tweets: 
    tweet = str(tweet[0])
    bag_of_words = get_bag_of_words(tweet, vocab)
    train_bag_of_words.append(bag_of_words)

for i in range(10): 
    print(train_tweets[i])
    print(list(train_bag_of_words[i].values()))


# In[111]:


# Bag of words for test data 
test_tweets = test.values
test_bag_of_words = [] 
for tweet in test_tweets: 
    tweet = str(tweet[0])
    bag_of_words = get_bag_of_words(tweet, vocab)
    test_bag_of_words.append(bag_of_words)

for i in range(10): 
    print(test_tweets[i])
    print(list(test_bag_of_words[i].values()))

