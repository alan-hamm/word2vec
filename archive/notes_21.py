

# loading in all the essentials for data manipulation
import pandas as pd
import numpy as np

# collections is a built in module, there are 6 most commonaly used data 
#stuctures in collection modules e.g. someof the classes are  Defaultdic, Counter 
import collections

#load in the NTLK stopwords to remove articles, preposition and other words that are not actionable

from nltk.corpus import stopwords
# This allows to create individual objects from a bag of words
# nltk package contains a module called tokenize() and this futher conatins 
# word_tokenize method 
from nltk.tokenize import word_tokenize
# Lemmatizer helps to reduce words to the base form
from nltk.stem import WordNetLemmatizer
# Ngrams allows to group words in common pairs or trigrams..etc
from nltk import ngrams

# visual library
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer

df=pd.read_sas('q4_rspntoth_nodup.sas7bdat',encoding='latin-1')
    
df.info()
df.head(100)

with pd.option_context('display.max_rows',None, 'display.max_columns', None, 'display.width',100):
    print(df.head(100))
    
df.info()
    

with pd.option_context('display.max_columns', None):
    print(df.head(100)) 

df.isnull().sum()
df.info()

def word_frequency(chi):
    # joins all the sentenses
    #f7notes_str = str(f7notes)
    chi2 =''.join(chi)
   # f7notes23 = f7notes2.replace('b''', ' ') 
    # creates tokens, creates lower class, removes numbers and lemmatizes the words
    new_tokens = word_tokenize(chi2)
    new_tokens = [t.lower() for t in new_tokens]
    new_tokens =[t for t in new_tokens if t not in stopwords.words('english')]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    new_tokens =[lemmatizer.lemmatize(t) for t in new_tokens]
    #counts the words, pairs and trigrams
    counted = collections.Counter(new_tokens)
    counted_2= collections.Counter(ngrams(new_tokens,2))
    counted_3= collections.Counter(ngrams(new_tokens,3))
    #creates 3 data frames and returns thems
    word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
    word_pairs =pd.DataFrame(counted_2.items(),columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
    trigrams =pd.DataFrame(counted_3.items(),columns=['trigrams','frequency']).sort_values(by='frequency',ascending=False)
    return word_freq,word_pairs,trigrams

data2, data3, data4 = word_frequency(df)

# create subplot of the different data frames
fig, axes = plt.subplots(3,1,figsize=(8,20))
sns.barplot(ax=axes[0],x='frequency',y='word',data=data2.head(30))
sns.barplot(ax=axes[1],x='frequency',y='pairs',data=data3.head(30))
sns.barplot(ax=axes[2],x='frequency',y='trigrams',data=data4.head(30))


def word_frequency(chi):
    # joins all the sentenses
    #f7notes_str = str(f7notes)
    chi2 =''.join(chi)
   # f7notes23 = f7notes2.replace('b''', ' ') 
    # creates tokens, creates lower class, removes numbers and lemmatizes the words
    new_tokens = word_tokenize(chi2)
    new_tokens = [t.lower() for t in new_tokens]
    new_tokens =[t for t in new_tokens if t not in stopwords.words('english')]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    new_tokens =[lemmatizer.lemmatize(t) for t in new_tokens]
    #counts the words, pairs and trigrams
    counted = collections.Counter(new_tokens)
    counted_2= collections.Counter(ngrams(new_tokens,2))
    counted_3= collections.Counter(ngrams(new_tokens,3))
    #creates 3 data frames and returns thems
    word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
    word_pairs =pd.DataFrame(counted_2.items(),columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
    trigrams =pd.DataFrame(counted_3.items(),columns=['trigrams','frequency']).sort_values(by='frequency',ascending=False)
    return word_freq,word_pairs,trigrams

data2, data3, data4 = word_frequency(df_vis['PRSPNDOTH'])

# create subplot of the different data frames
fig, axes = plt.subplots(3,1,figsize=(8,20))
sns.barplot(ax=axes[0],x='frequency',y='word',data=data2.head(30))
sns.barplot(ax=axes[1],x='frequency',y='pairs',data=data3.head(30))
sns.barplot(ax=axes[2],x='frequency',y='trigrams',data=data4.head(30))

#Maybe derving the opinion or the attiuded of the FR ??
# Importing SentimentIntensityAnalyzer class from nltk.sentiment module
# And use polarity_scores method from SentimentIntensityAnalyzer Class

sia = SentimentIntensityAnalyzer()
sia.polarity_scores('Today is a very good day..')
