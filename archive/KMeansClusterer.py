# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:06:09 2022

@author: pqn7
"""

import nltk
from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin
#from nltk.corpus import wordnet as wn
import unicodedata
from nltk.stem import WordNetLemmatizer
        
#ETL SAS dataset
corpus = nltk.corpus.reader.plaintext.PlaintextCorpusReader(".", 'y19_y22_rspntoth.txt')
       
class KMeansClusters(BaseEstimator, TransformerMixin):
    def __init__(self,k=7):
        """
        k is the number of clusters
        model is the implementation of Kmeans
        """
        
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance, avoid_empty_clusters=True)
        
    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents):
        """
        Fits the K-Means model to one-hot vectorized documents
        """
        return self.model.fit_predict(documents)
        
class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, language='english'):
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
    def is_punct(self, token):
        return all(unicodedata.category(char).startswith('P') for char in token)
        
    def is_stopword(self, token):
        return token.lower() in self.stopwords
    
    def normalize(self, document):
        return[
                self.lemmatize(token).lower()
                for paragraph in document
                for sentence in paragraph
                for token in sentence
                if not self.is_punct(token) and not self.is_stopword(token)
            ]
            
    def lemmatize(self, token):
       return self.lemmatizer.lemmatize(token)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, documents):
        return [' '.join(self.normalize(doc)) for doc in documents]      
            
from sklearn.feature_extraction.text import CountVectorizer
class OneHotVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer(binary=True)
        
    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents):
        freqs = self.vectorizer.fit_transform(documents)
        return [freq.toarray()[0] for freq in freqs]
    
from sklearn.pipeline import Pipeline

docs = corpus.paras()
model = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', OneHotVectorizer()),
        ('clusters', KMeansClusters(k=7))
        ])
                
clusters = model.fit_transform(docs)
for idx, cluster in enumerate(clusters):
    print(cluster)
        