# -*- coding: utf-8 -*-
"""
Script to execute unsupervised learning of NHIS verbatim fields. Includes: case notes,
contact history instrument, and F7notes.

Created on Fri Sep 16 2022

@author: pqn7
"""

import pandas as pd
import numpy as np
import nltk, string

#ETL SAS dataset
corpus = nltk.corpus.reader.plaintext.PlaintextCorpusReader(".", 'y19_y22_rspntoth.txt')

''' parse for paragraphs '''
paras=corpus.paras()
#for p in paras:
#    print(p)
    
''' parse for sentences '''
sent=corpus.sents()
#for s in sent:
#    print(s)
    
''' parse for words '''
words=corpus.words()
for w in words:
    print(w)
    
 
#ONE-HOT ENCODING for vectorization  
 
# Create an empty dictionary
token_index = {}
#Create a counter for counting the number of key-value pairs in the token_length
counter = 0

# Select the elements of the samples which are the two sentences
for s in sent:                                      
  for considered_word in str(s).split():
    if considered_word not in token_index:
      
      # If the considered word is not present in the dictionary token_index, add it to the token_index
      # The index of the word in the dictionary begins from 1 
      token_index.update({considered_word : counter + 1}) 
      
      # updating the value of counter
      counter = counter + 1                                

print(token_index)


# Create a tensor of dimension 3 named results whose every elements are initialized to 0

results  = np.zeros(shape = (len(sent),
                            len(sent),
                            max(token_index.values()) + 1))  
print(results.shape)
#print(results)

# Now create a one-hot vector corresponding to the word
# iterate over enumerate(samples) enumerate object
for i, s in enumerate(sent):
    # Convert enumerate object to list and iterate over resultant list 
    for j, considered_word in list(enumerate(str(s).split())):
        # set the value of index variable equal to the value of considered_word in token_index
        index = token_index.get(considered_word)
    
        # In the previous zero tensor: results, set the value of elements with their positional index as [i, j, index] = 1.
        results[i, j, index] = 1.
print(results)



