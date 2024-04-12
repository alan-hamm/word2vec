from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from readability.readability import Unparseable
from readability import Document as Paper
import codecs, os, time, nltk
from bs4 import BeautifulSoup
from nltk import sent_tokenize, word_tokenize, pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import KeyedVectors
from gensim.corpora.dictionary import Dictionary


cat_pattern = r'(a-z_\s]+)/.'
doc_pattern = r'[\w\d_]+\.html'

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
 
                 
tags = ['h1','h2','h3','h4','h5','h6','h7','p','li','td'] 
corpus = CIPSEACorpusReader(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\corpora\rspndoth")
#corpus = corpus.paras()

from gensim.models.doc2vec import TaggedDocument, Doc2Vec

_corpus=[]
for p in corpus.sents():
    _corpus.append(p)

corpus = [_corpus for doc in _corpus]
corpus = [
        TaggedDocument(words, ['d{}'.format(idx)])
        for idx, words in enumerate(corpus)
        ]
model = Doc2Vec(corpus, size=5, min_count=0)
print(model.docvecs[0])
#model.save(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\models\rspndoth_doc2vec.model")

import gensim
from gensim.matutils import sparse2full

_corpus = CIPSEACorpusReader(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\corpora\rspndoth")
corpus = []
for w in _corpus.words():
    corpus.append(w)
corpus = [corpus]
print(corpus)

lexicon = gensim.corpora.Dictionary(corpus)
docvec = KeyedVectors.load(r'\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\models\rspndoth_doc2vec.model', mmap='r')
vectors = docvec.docvecs[0]

lexicon.save_as_text('lexicon.txt', sort_by_word=True)




corpus = CIPSEACorpusReader(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\corpora\rspndoth")

import unicodedata
from nltk.stem import WordNetLemmatizer
        
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
                self.lemmatizer.lemmatize(corpus.words())
                for paragraph in document
                for sentence in paragraph
                for word in sentence
                if not self.is_punct(word) and not self.is_stopword(word)
            ]
    
    def lemmatize(self, word):
        self.lemmatizer = WordNetLemmatizer()
        return self.lemmatizer
    
    def fit(self, documents, labels=None):
        return self
        
    def transform(self, documents):
        for document in documents:
            yield self.normalize(document)


class GensimDoc2VecVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, dirpath='.', tofull=False):
        self._lexicon_path = os.path.join(dirpath, "corpus.dict")
        self._docvec_path = os.path.join(dirpath, "docvec.model")
        
        self.lexicon = None
        self.docvec = None
        self.tofull = tofull
        
        self.load()
        
    def load(self):
        if os.path.exists(self._lexicon_path):
            self.lexicon = Dictionary.load(self._lexicon_path)
            
        if os.path.exists(self._docvec_path):
            self.docvec = Doc2Vec().load(self._docvec_path)
    
    def save(self):
        self.lexicon.save(self._lexicon_path)
        self.docvec.save(self._docvec_path)
        
    def fit(self, documents, labels=None):
        self.lexicon = Dictionary(documents)
        self.docvec = Doc2Vec([
                self.lexicon.doc2bow(doc)
                for doc in documents],
                id2word=self.lexicon)
        self.save()
        return self
    
    def transform(self, documents):
        def generator():
            for document in documents:
                vect = self.docvec[self.lexicon.doc2bow(document)]
                if self.tofull:
                    yield sparse2full(vect)
                else:
                    yield vect
        return list(generator())
    
from sklearn.pipeline import Pipeline
from gensim.sklearn_api import ldamodel

class GensimTopicModels(object):
    def __init__(self, n_topics=50):
        self.n_topics = n_topics
        self.model = Pipeline([
                ('norm', TextNormalizer()),
                ('vect', GensimDoc2VecVectorizer()),
                ('model', ldamodel.LdaTransformer(num_topics = self.n_topics))
                ])
    
    def fit(self, documents):
        self.model.fit(documents)
        return self

    
    def get_topics(vectorized_corpus, model):
        from operator import itemgetter
        
        topics = [
                max(model[doc], key = itemgetter(1))[0]
                for doc in vectorized_corpus
                ]
        return topics
    

docs = [
        list(corpus.docs(fileids=fileid))[0]
        for fileid in corpus.fileids()
        ]


import pyLDAvis
import pyLDAvis.gensim

gensim_lda=GensimTopicModels()

lda = gensim_lda.model.named_steps['model'].gensim_model

_corpus = CIPSEACorpusReader(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\corpora\rspndoth")
corpus = []
for w in _corpus.words():
    corpus.append(w)
corpus = [corpus]
print(corpus)

lexicon = gensim.corpora.Dictionary(corpus)
docvec = KeyedVectors.load(r'\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\models\rspndoth_doc2vec.model', mmap='r')
vectors = docvec.docvecs[0]
for _t in lda.get_topics(vectors):
    print(_t)
data=pyLDAvis.gensim.prepare(docvec,corpus,lexicon)
pyLDAvis.display(data)