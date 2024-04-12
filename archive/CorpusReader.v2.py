'''
author alan hamm(pqn7)
date september 2022

script to perform text analysis of PII_Master case notes and
contact history instrument
'''

from pprint import pprint# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing
#from nltk.util import ngrams
#import spacy# Plotting tools
import en_core_web_lg
#import pyLDAvis
#import pyLDAvis.gensim
#import matplotlib.pyplot as plt

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from readability.readability import Unparseable
from readability import Document as Paper
import codecs, os, time, nltk
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
stop_words = stopwords.words('english')
#from sklearn.base import BaseEstimator, TransformerMixin
#from gensim.models import KeyedVectors
#from gensim.corpora.dictionary import Dictionary


cat_pattern = r'(a-z_\s]+)/.*'
doc_pattern = r'[\w\d_]+\.html'
tags = ['h1','h2','h3','h4','h5','h6','h7','p','li','td']
class CIPSEACorpusReader(CategorizedCorpusReader, CorpusReader):
    def __init__(self, root, fileids=doc_pattern, encoding='latin=1', **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = cat_pattern
            
        CategorizedCorpusReader.__init__(self,kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)
        
    def resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")
            
        if categories is not None:
            return self.fileids(categories)
        return fileids
    
    def docs(self,fileids=None, categories=None):
        fileids = self.resolve(fileids, categories)
        
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield f.read()
    
    def sizes(self, fileids=None, categories=None):
        fileids = self.resolve(fileids, categories)
        
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

    def html(self, fileids=None, categories=None):
        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
                continue
            
    def paras(self, fileids=None, categories=None):
        for html in self.html(fileids, categories):
            soup=BeautifulSoup(html,'lxml')
            for element in soup.find_all(tags):
                yield element.text
            soup.decompose()
    
    def to_raw(self, fileids=None, categories=None):
        raw = ' '
        for paragraph in self.paras(fileids, categories):
            raw += ' ' + paragraph
        yield raw
            
                
    def sents(self, fileids=None, categories=None):
        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence
    
    def words(self, fileids=None, categories=None):
        for sentence in self.sents(fileids, categories):
            for token in word_tokenize(sentence):
                yield token
                
    def tokenize(self, fileids=None, categories=None):
        for paragraph in self.paras(fileids, categories):
            yield[
                    pos_tag(word_tokenize(sent))
                    for sent in sent_tokenize(paragraph)
                    ]
    
    def describe(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and
        returns a dictionary with a variety of metrics
        concerning the state of the corpus.
        """
        started = time.time()
        
        #structures to perform counting
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()
        
        #perform single pass over paragraphs, tokenize and count
        for para in self.paras():
            counts['paras'] += 1
            
            for sent in para:
                counts['sents'] += 1
                
                for word in sent:
                    counts['words'] += 1
                    tokens[word] += 1
                    
        #compute  the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))
        
        #return data structure with information
        return{
                'files': n_fileids,
                'topics': n_topics,
                'paras': counts['paras'],
                'sents': counts['sents'],
                'words': counts['words'],
                'vocab': len(tokens),
                'lexdiv': float(['words']) / float(len(tokens)),
                'ppdoc': float(counts['paras']) / float(n_fileids),
                'sppar': float(counts['sents']) / float(counts['paras']),
                'secs': time.time() - started,
                }


''' Read corpus '''
_corpus = CIPSEACorpusReader(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\corpora")
corpus = []
for w in _corpus.sents():
    corpus.append(w)
#pprint(corpus)

''' sentiment analysis '''
'''
corpus_raw = ' '
for sentence in _corpus.paras():
    corpus_raw += sentence
pprint(corpus_raw)


from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

for sentence in _corpus.sents():
    print(sentence)
    t=sia.polarity_scores(sentence)
    print(t)
'''

    
''' 
function to parse words within sentence
creates a list of a list of words within a sentence
'''
def sent_to_words(sentences):
    for sentence in sentences:
        #yield sentence 
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) #deacc True removes punctuation


''' build n-gram models '''
'''
#print n-gram, collocation, wordcloud
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
bi_dict=dict()
bg_measures= BigramAssocMeasures()

for t in data_words_nostops:
    words = _corpus.words()
    bi_finder = BigramCollocationFinder.from_words(words)
    bi_collocs = bi_finder.nbest(bg_measures.likelihood_ratio, 10)
    
    for colloc in bi_collocs:
        bi_dict[colloc] += 1
'''

tokenizer = RegexpTokenizer(r'(\w+)')
unigram = []
bigram=[]
trigram=[]
fourgram=[]
tokenized_text = []
#print(data_words)
for sentence in sent_to_words(corpus):
    sentence = str(sentence).lower()
    sequence = tokenizer.tokenize(sentence)
    for word in sequence:
        if (word == '.'):
            sequence.remove(word)
        else:
            unigram.append(word)
    tokenized_text.append(sequence)
    bigram.extend(list(ngrams(sequence, 2)))
    trigram.extend(list(ngrams(sequence, 3)))
    fourgram.extend(list(ngrams(sequence, 4)))

def removal(x):
    y = []
    for pair in x:
        count = 0
        for word in pair:
            if word in stop_words:
                count = count or 0
            else:
                count = count or 1
        if (count ==1):
            y.append(pair)
    return(y)

bigram = removal(bigram)
trigram = removal(trigram)
fourgram = removal(fourgram)

freq_bi = nltk.FreqDist(bigram)
freq_tri = nltk.FreqDist(trigram)
#pprint(freq_bi)
freq_four = nltk.FreqDist(fourgram)

for t in freq_bi.most_common(50):
    print("50 Most common bigrams: ", t)
for t in freq_tri.most_common(50):
    print("50 Most common trigrams: ", t)
for t in freq_four.most_common(50):
    print("50 Most common fourgrams: ", t)


''' 
topic profiling 
'''
nlp = en_core_web_lg.load( disable=['parser','ner'])
# define function for stopwords, bigrams, trigrams, and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out=[]
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data_words = list(sent_to_words(corpus))
data_words = remove_stopwords(data_words)
#print(data_words)
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5,
                               threshold=100) # higher threshold fewer phrases
trigram = gensim.models.Phrases(bigram[data_words],
                                threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
#pprint(trigram_mod[bigram_mod[data_words[0]]])
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)
data_words_trigrams = make_trigrams(data_words_bigrams)
#print(data_words_bigrams)

bigram_nostops = remove_stopwords(data_words_bigrams)
data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#pprint(data_lemmatized[0:25])


id2word = corpora.Dictionary(data_lemmatized)
texts = data_lemmatized
_corpus = [id2word.doc2bow(text) for text in texts]

#remove empty lists from corpus
corpus=[]
for c in _corpus:
    if c != []:
        corpus.append(c)
#pprint(corpus[0:25])

[[(id2word[id], freq) for id, freq in cp] for cp in corpus[0:25]]
print(id2word) # print dictionary metadata(incl. unique token count)
#print(id2word.token2id) #print tokens along with their ID

#model = KeyedVectors.load(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\models\allvarsShuffled_doc2vec.model", mmap='r')
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=25,
                                            random_state=1000, # this serves as a seed, and in case wanted to repeat exactly the traning process
                                            update_every=0, # update the model every update_every chucksize chunks(essentially this is for memory consumption optimization
                                            chunksize=5, # number of documents to be used in each training chunk
                                            passes=25, # number of documents to be iterated through for each update. Set to O for batch learning, > 1 for online iterative learning
                                            alpha='auto',
                                            decay = .5, # A number between (0.5, 1] to weight what percentage of the prev lambda value is forgotten when each new document is examined
                                            minimum_probability = .1, #topics with a probability lower than this threshold will be filtered out
                                            per_word_topics=True) # setting to True allows for extraction of the most likely topics given a word


#save model to disk
#from gensim.test.utils import datapath
#tmp_file = datapath(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\models\allvarsShuffled.model")
#lda_model.save(tmp_file)

# Print the keyword of topicsp
pprint(lda_model.print_topics())
lda_model.show_topic(1)


for topic_id in range(lda_model.num_topics):
    topk = lda_model.show_topic(topic_id, 7)
    topk_words = [ w for w, _ in topk ]
    topics_string=[]
    topics_string.append(topic_id)
    print(topk_words)
    #print('{}: {}'.format(topic_id, ' '.join(topk_words)))



'''
Lexical Dispersion Plot
    ...a measure of how frequently a word appears across the parts of 
    a corpus. The plot notes the occurrences of a word and how many words from the beginning
    of the corpus it appears(word offsets).
'''
from nltk.text import Text
from nltk.draw.dispersion import dispersion_plot
import matplotlib.pyplot as plt

_corpus = CIPSEACorpusReader(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\corpora")
corpus = ' '.join(str(e) for e in _corpus.words())
#print(corpus)
tokens = wordpunct_tokenize(corpus)
tokens_text = Text(tokens)

#rspndoth
targets = ['XXX']
plt.figure(figsize=(15,8))
dispersion_plot(tokens_text, targets, ignore_case=True, title='Lexical Dispersion Plot')






'''
# compute perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))

# compute coherence score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
'''




