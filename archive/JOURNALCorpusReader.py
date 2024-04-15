'''
author: alan hamm(pqn7)
date march 2024

resources:
    Applied Text Analysis with Python by Bengfort et al.

script to perform text analysis on CDC Journel Harvest

'''

#%%

# https://www.nltk.org/
# https://www.nltk.org/api/nltk.corpus.reader.api.html#nltk.corpus.reader.api.CategorizedCorpusReader
# https://www.nltk.org/api/nltk.corpus.reader.api.html#nltk.corpus.reader.api.CorpusReader
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
import nltk.data
from nltk import sent_tokenize, word_tokenize, pos_tag, wordpunct_tokenize
# sent_tokenize: https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.sent_tokenize
# word_tokenize: https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.word_tokenize
# pos_tag: https://www.nltk.org/api/nltk.tag.pos_tag.html


''' 
topic profiling 
'''
import en_core_web_lg
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing
from nltk.corpus import stopwords

# https://github.com/buriy/python-readability
from readability.readability import Unparseable
from readability.readability import Document as Paper

# https://docs.python.org/3/library/time.html
import time

# https://beautiful-soup-4.readthedocs.io/en/latest/
import bs4

# https://docs.python.org/3/library/codecs.html
import codecs

# https://docs.python.org/3/library/json.html
import json

import re 

import os

import pprint as pp

import string

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from time import time  # To time our operations

nlp = en_core_web_lg.load( disable=['parser','ner'])

stop_words = stopwords.words('english')


#%%
# we create a list to contain the json files that
# are to be processed
#DOC_ID = ['mmsu_art_en_html.json']
DOC_ID=[]
for x in os.listdir(r"C:/_harvester/data/html-by-year"):
    if x.endswith(".json") and x[:4] in ('2020'):
        DOC_ID.append(x)
print(DOC_ID)
#DOC_ID =r'([\d+]_html\.json)'

# we create a list of categories/keywords/tags to
# be used to refine searches
# CAT_PATTERN = r'([0-9]+\.htm$)'
CAT_PATTERN =r'([\d]+)_html\.json'

# we mark the HTML tags to be used for 
# extacting the desired article, etc. text
TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'p', 'li']
#TAGS = ['h1']


#%%

class JOURNALCorpusReader(CategorizedCorpusReader, CorpusReader):
    """ a corpus reader for CDC Journal articles """
    # class nltk.corpus.reader.api.CorpusReader
    # we explicitly specify the encoding as utf8 even though
    # the default is utf8
    def __init__(self, root, tags=TAGS, fileids=DOC_ID, encoding='utf8', **kwargs):
            
        # we use this check to see if the user specified any
        # values in the CAT_PATTERN list
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        # initialize the NLTK  reader objects
        # review https://www.nltk.org/api/nltk.corpus.reader.api.html#nltk.corpus.reader.api.CategorizedCorpusReader to see
        # how __init__ is defined for each module; for the categorized
        # corpus reader, we use it to create categories if none are specified.
        CategorizedCorpusReader.__init__(self, kwargs)

        # https://www.nltk.org/api/nltk.corpus.reader.api.html#nltk.corpus.reader.api.CorpusReader
        # encoding –
        # The default unicode encoding for the files that make up the corpus. The value of encoding can 
        # be any of the following:
        #   A string: encoding is the encoding name for all files.
        #   A dictionary: encoding[file_id] is the encoding name for the file whose identifier is file_id. If file_id is not in encoding, 
        #       then the file contents will be processed using non-unicode byte strings.
        #   A list: encoding should be a list of (regexp, encoding) tuples. The encoding for a file whose 
        #       identifier is file_id will be the encoding value for the first tuple whose regexp matches
        #        the file_id. If no tuple’s regexp matches the file_id, the file contents will be processed using non-unicode byte strings.
        #   None: the file contents of all files will be processed using non-unicode byte strings.
        CorpusReader.__init__(self, root, fileids, encoding)
        
        self.fileids = fileids
        #self.categories = self.categories()
        self.tags = tags

        #print("From the constructor these are the fileids", fileids)
        #print("from the constructor these are the categories", self.categories)
        
        

    # we create a method that will allow us to filter how we
    # read the data from disk, either by specifying a list of categories
    # or a list of filenames
    def resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
           raise ValueError("Specify fileids or categories, not both")
            
        if categories is not None:
            #pp.pprint("This is a test of the resolve() method where categories is not None:", self.categories)
            return self.fileids(categories)
    
        #pp.pprint("This is a test of the resolve() method where categories IS None:", self.categories)
        return fileids

    # we use this method to read all values from the key-value objects,
    # concatenating them into a list object which is returned.
    def docs(self,fileids=None, categories=None):

        fileids = self.resolve(fileids, categories)
        #for f in fileids:
        #    pp.pprint("This is a list of the fileids in doc():", f)
        
        # https://docs.python.org/3/library/codecs.html
        # This module defines base classes for standard Python codecs 
        # (encoders and decoders) and provides access to the internal Python 
        # codec registry, which manages the codec and error handling 
        # lookup process. Most standard codecs are text encodings, which encode 
        # text to bytes (and decode bytes to text), but there are also codecs 
        # provided that encode text to text, and bytes to bytes. Custom codecs 
        # may encode and decode between arbitrary types, but some module features 
        # are restricted to be used specifically with text encodings or with codecs 
        # that encode to bytes.

        # A string in Python is a sequence of Unicode code points (in range U+0000–U+10FFFF). To store or 
        # transfer a string, it needs to be serialized as a sequence of bytes.
        # Serializing a string into a sequence of bytes is known as “encoding”, 
        # and recreating the string from the sequence of bytes is known as “decoding”.
        # There are a variety of different text serialization codecs, which are collectively 
        # referred to as “text encodings”.

        # codecs.open(filename, mode='r', encoding=None, errors='strict', buffering=-1)
        # Open an encoded file using the given mode and return an instance of StreamReaderWriter, 
        # providing transparent encoding/decoding. The default file mode is 'r', meaning to open 
        # the file in read mode.
        # Note If encoding is not None, then the underlying encoded files are always opened in binary 
        # mode. No automatic conversion of '\n' is done on reading and writing. The mode argument
        # may be any binary mode acceptable to the built-in open() function; 
        # the 'b' is automatically added.
        
        # abspaths() Return a list of the absolute paths for all fileids in this corpus; 
        #       or for the given list of fileids, if specified.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            #print("This is a test of the docs() method using the codecs module", path)

            with codecs.open(path, 'r', encoding=encoding) as f:
                data = json.load(f)
                #for key, value in data.items():
                #    json_list.append(value)

                #return data.values()
                return data

    # we use this method to iterate over each key-value pair, specifically
    # iterating over the value ie HTML content
    def html(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary() # summer() Given a HTML file, extracts the text of the article
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
                continue
            
    def paras(self, fileids=None, categories=None):
        for html in self.html(fileids, categories):
            soup=bs4.BeautifulSoup(html,'html.parser')
            for element in soup.find_all(TAGS):
                #if not any(c.isnumeric() for c in element.text):
                yield element.text
            soup.decompose()
                
    def sents(self, fileids=None, categories=None):
        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                if sentence.find('mmwrq@cdc.gov') == -1 and \
                   sentence.find('website') == -1 and \
                   sentence.find('web') == -1: 
                #    not any(c.isnumeric() for c in sentence):
                    yield sentence

    def words(self, fileids=None, categories=None):
        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token
    
    def tokenize(self, fileids=None, categories=None):
        for paragraph in self.paras(fileids, categories):
            yield[
                    pos_tag(wordpunct_tokenize(sent))
                    for sent in sent_tokenize(paragraph)
                    ]    
            
    def describe(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and
        returns a dictionary with a variety of metrics
        concerning the state of the corpus.
        """
        started = time()
        
        #structures to perform counting
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()
        
        #perform single pass over paragraphs, tokenize and count
        for para in self.paras():
            counts['paras'] += 1
            
        for sent in self.sents():
            counts['sents'] += 1
                
        for word in self.words():
            counts['words'] += 1
            tokens[word] += 1


        #compute  the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids)
        #n_fileids = len(fileids)
        #n_topics = len(self.categories(self.resolve(fileids, categories)) or self.categories)
        #n_topics = len(categories)
        
        #return data structure with information
        return{
                # number of files
                'files': n_fileids,
                # number of topics
                #'topics': n_topics,
                # number of paragraphs
                'paras': counts['paras'],
                # number of sentences
                'sents': counts['sents'],
                # number of words
                'words': counts['words'],
                # size of vocabulary ie number of unique terms
                'vocab': len(tokens),
                # lexical diversity, the ratio of unique terms to total words
                'lexdiv': float(counts['words']) / float(len(tokens)),
                # average number of paragraphs per document
                'ppdoc': float(counts['paras']) / float(n_fileids),
                # average number of sentences per paragraph
                'sppar': float(counts['sents']) / float(counts['paras']),
                # total processing time
                'secs': time() - started,
                }

#%%
_corpus = JOURNALCorpusReader('C:/_harvester/data/html-by-year')


#%%

_corpus.describe()



#%%

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

w2v_model = Word2Vec(min_count=20, # (int, optional) – Ignores all words with total frequency lower than this.
                     window=2, # Maximum distance between the current and predicted word within a sentence.
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=15,
                     workers=cores-1)



#%%

t = time()
#sentences =[x for x in _corpus.sents()]
sentences = []

word = ''
for w in _corpus.words():
    #sentences.append(re.sub(r'[^\w\s]', '', sent))
    if w not in string.punctuation and w.isnumeric() == False:
        word += w
    sentences.append(word)
    word = ''

#sentences = [x.split() for x in sentences]
sentences = [[word for word in sentences if word.lower() not in stop_words and len(word)> 2]]

print('Time to build sentences: {} mins'.format(round((time() - t) / 60, 2)))

pp.pprint(sentences[0:25])


#%%
t = time()
bigram = Phrases(sentences, min_count=5, threshold=10000) # higher threshold fewer phrases.

pp.pprint('Time to build bigram: {} mins'.format(round((time() - t) / 60, 2)))


#%%
for ngrams, _ in bigram.vocab.items():
    #unicode_ngrams = ngrams.decode('utf-8')
    if '_' in ngrams:
        print(ngrams)


#%%
t = time()

w2v_model.build_vocab(sentences, progress_per=100)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))


#%%
t = time()

print(sentences[0:5])
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))


#%%
w2v_model.init_sims(replace=True)


#%%
os.makedirs(f"C:/_harvester/data/word2vec-models/2020/")
w2v_model.save(f"C:/_harvester/data/word2vec-models/2020/word2vec-2020.model")


#%%
print(w2v_model.corpus_total_words)

#%%

similar_words = w2v_model.wv.most_similar(positive=["CDC"])
for word, similarity in similar_words:
    print(f"{word}: {similarity}")






#%%
import re, string 
import pandas as pd   
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from gensim.models import Word2Vec
from matplotlib import pyplot as plt

def tsne_plot(model):
    "Create TSNE model and plot it"
    labels = pd.DataFrame()
    tokens = pd.DataFrame()

    for word in list(model.wv.key_to_index):
        pp.pprint(word)
        tokens=pd.concat([tokens, pd.DataFrame(model.wv[word])])
        labels = pd.concat([labels, pd.DataFrame(model.wv[word])])
    
    tsne_model = TSNE(perplexity=40, n_components=1, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(18, 18)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
   
tsne_plot(w2v_model)



#%%


def w2v_to_numpy (model):
  """ Convert the word2vec model (the embeddings) into numpy arrays.
  Also create and return the mapping of words to the row numbers.

  Parameters:
  ===========
  model (gensim.Word2Vec): a trained gensim model

  Returns:
  ========
  embeddings (numpy.ndarray): Embeddings of each word
  idx, iidx (tuple): idx is a dictionary mapping word to row number
                     iidx is a dictionary mapping row number to word
  """
  model.wv.init_sims()
  embeddings = deepcopy (model.wv.get_normed_vectors())
  idx = {w:i for i, w in enumerate (model.wv.index_to_key)}
  iidx = {i:w for i, w in enumerate (model.wv.index_to_key)}
  return embeddings, (idx, iidx)


#%%
def near_neighbors (embs, query, word2rownum, rownum2word, k=5):
  """ Get the `k` nearest neighbors for a `query`

  Parameters:
  ===========
  embs (numpy.ndarray): The embeddings.
  query (str): Word whose nearest neighbors are being found
  word2rownum (dict): Map word to row number in the embeddings array
  rownum2word (dict): Map rownum from embeddings array to word
  k (int, default=5): The number of nearest neighbors

  Returns:
  ========
  neighbors (list): list of near neighbors;
                    size of the list is k and each item is in the form
                    of word and similarity.
  """

  sims = np.dot (embs, embs[word2rownum[query]])
  indices = np.argsort (-sims)
  return [(rownum2word[index], sims[index]) for index in indices[1:k+1]]


#%%

_2019_model = Word2Vec.load(r"C:\_harvester\data\word2vec-models\2019\word2vec-2019.model")
_2020_model = Word2Vec.load(r"C:\_harvester\data\word2vec-models\2020\word2vec-2020.model")

#%%
_19_embs, (_19_idx, _19_iidx) = w2v_to_numpy (_2019_model)
_20_embs, (_20_idx, _20_iidx) = w2v_to_numpy (_2020_model)


#%%
query = 'virus'
print (f'Near neighbors for {query} in the 2019 corpus')
for item in near_neighbors (_19_embs, query, _19_idx, _19_iidx, k=10):
  print (item)
print ()
print (f'Near neighbors for {query} in the 2020 corpus')
for item in near_neighbors (_20_embs, query, _20_idx, _20_iidx, k=10):
  print (item)









# %%
''' 
topic profiling 
'''

t = time()
word = ''
sentences = []
for w in _corpus.words():
    #sentences.append(re.sub(r'[^\w\s]', '', sent))
    if w not in string.punctuation and w.isnumeric() == False:
        word += w
    sentences.append(word)
    word = ''

data_words = [[word for word in _corpus.words() if word.lower() not in stop_words and len(word)> 2]]


pp.pprint('Time to build word list: {} mins'.format(round((time() - t) / 60, 2)))

pp.pprint(data_words)


#%%

#print(data_words)
# Build the bigram and trigram models
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

#pp.pprint(trigram_mod[bigram_mod[data_words[1]]])
# Faster way to get a sentence clubbed as a trigram/bigram
#bigram_mod = gensim.models.phrases.Phraser(bigram)
#trigram_mod = gensim.models.phrases.Phraser(trigram)


#%%
#for ngrams in bigram_mod:
for ngrams, _ in bigram.vocab.items():
    #unicode_ngrams = ngrams.decode('utf-8')
    if '_' in ngrams:
        print(ngrams)

        
#%%

def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    #texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [[word for word in texts if str(word).lower() not in stop_words]]

    texts = [bigram_mod[doc] for doc in texts]
    
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    

    nlp = en_core_web_lg.load( disable=['parser','ner'])
    nlp.max_length = 50000000

    for sent in texts:
        for word in sent:
            doc = nlp(" ".join(word)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in doc if word.lower() not in stop_words] for doc in texts_out]    
    return texts_out


#%%
t = time()
data_ready = process_words(data_words)  # processed Text Data!

pp.pprint('Time to process  word list: {} mins'.format(round((time() - t) / 60, 2)))

#%%
pp.pprint(data_ready[0:25])


#%%
# Create Dictionary
id2word = corpora.Dictionary(data_ready)


#%%
# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]


#%%
# Build LDA model
from gensim.test.utils import datapath
for x in os.listdir(r"C:/_harvester/data/html-by-year"):
    if x.endswith(".json") and x[:4] in ('2020'):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=25, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

        os.makedirs(f"C:/_harvester/data/lda-models/{x}")
        fname = datapath(f"C:/_harvester/data/lda-models/{x}/_{x[:4]}")
        lda_model.save(fname)

pp.pprint(lda_model.print_topics())
pp.pprint(lda_model.show_topic(1))



#%%
import pandas as pd
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data_ready):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = pd.concat([sent_topics_df,pd.Series([int(topic_num), round(prop_topic,4), topic_keywords])], ignore_index=True)
            else:
                break
    #sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
#df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)








#%%


























#%%
trigram_nostops = remove_stopwords(trigram)

#%%
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
pp.pprint(data_lemmatized[0:25])


#%%
data_lematized_trunc = [x for x in data_lemmatized if len(x) > 3]
pp.pprint(data_lematized_trunc[0:25])


#%%
id2word = corpora.Dictionary(data_lematized_trunc)
texts = data_lemmatized
_corpus = [id2word.doc2bow(text) for text in texts]

#%%

#remove empty lists from corpus
corpus=[]
for c in _corpus:
    if c != []:
        corpus.append(c)
pp.pprint(corpus[0:25])


#%%

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[0:25]]
pp.pprint(id2word) # print dictionary metadata(incl. unique token count)
pp.pprint(id2word.token2id) #print tokens along with their ID



#%%
#model = KeyedVectors.load(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\models\allvarsShuffled_doc2vec.model", mmap='r')
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=50,
                                            random_state=1000, # this serves as a seed, and in case wanted to repeat exactly the traning process
                                            update_every=0, # update the model every update_every chucksize chunks(essentially this is for memory consumption optimization
                                            chunksize=5, # number of documents to be used in each training chunk
                                            passes=0, # number of documents to be iterated through for each update. Set to O for batch learning, > 1 for online iterative learning
                                            alpha='auto',
                                            decay = .5, # A number between (0.5, 1] to weight what percentage of the prev lambda value is forgotten when each new document is examined
                                            minimum_probability = .1, #topics with a probability lower than this threshold will be filtered out
                                            per_word_topics=True) # setting to True allows for extraction of the most likely topics given a word


#%%
pp.pprint(lda_model.print_topics())
pp.pprint(lda_model.show_topic(1))
#%%