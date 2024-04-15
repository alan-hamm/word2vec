
#%%

#import requests 
from pprint import pprint 
import io
from nltk.tokenize import sent_tokenize, word_tokenize


#%%
ff = io.open(r'C:\Users\pqn7\OneDrive - CDC\a.Reference.Documentation\GenAIGuidance-processed.txt', 'r', encoding='utf-8')
doc=ff.read()
ff.close()

#%%
from time import time
start = time()
for line in doc:
    sentences = " ".join([line for line in doc.split(" ")])
print(len(sentences))
pprint(sentences)

finish_time = round((time() - start)/60, 2)
print(f"Time to complete {finish_time}")

#%%
'''
We already have a sentence tokenizer, so we just need 
to run the sent_tokenize() method to create the array of sentences.
'''
# 1 Sentence Tokenize
sentences = sent_tokenize(sentence)
total_documents = len(sentences)
pprint(sentences[:10])


#%%
# 2 Create the Frequency matrix of the words in each sentence.
freq_matrix = _create_frequency_matrix(sentences)
pprint(freq_matrix)

#%%
'''
Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
'''
# 3 Calculate TermFrequency and generate a matrix
tf_matrix = _create_tf_matrix(freq_matrix)
pprint(tf_matrix)


#%%
# 4 creating table for documents per words
count_doc_per_words = _create_documents_per_words(freq_matrix)
pprint(count_doc_per_words)

'''
Inverse document frequency (IDF) is how unique or rare a word is.
'''
# 5 Calculate IDF and generate a matrix
idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
print(idf_matrix)


#%%

# 6 Calculate TF-IDF and generate a matrix
tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
pprint(tf_idf_matrix)



#%%



# 7 Important Algorithm: score the sentences
sentence_scores = _score_sentences(tf_idf_matrix)
pprint(sentence_scores)



#%%



# 8 Find the threshold
threshold = _find_average_score(sentence_scores)
pprint(threshold)

#%%
# 9 Important Algorithm: Generate the summary
summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
pprint(summary)

# %%
