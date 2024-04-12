from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
#from nltk.corpus import wordnet as wn
import unicodedata
#from nltk.tag import pos_tag
 
#ETL SAS dataset
corpus = nltk.corpus.reader.plaintext.PlaintextCorpusReader(".", 'rspntoth_2022q1.txt')

''' parse for paragraphs '''
paras=corpus.paras()
#for p in paras:
#    print(p)

''' parse for sentences '''
sent=corpus.sents()
#print(len(sent))
#for s in sent:
#    print(s)
    
''' parse for words '''
words=corpus.words()
#for w in words:
#    print(w)
    
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
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, documents):
        for sentence in documents.sents():
            return sentence

class SklearnTopicModels(object):
    def __init__(self, n_topics=10):
        """
        n_topics is the desired number of topics
        """
        self.n_topics = n_topics
        self.model = Pipeline([
                ('norm', TextNormalizer()),
                ('vect', CountVectorizer(preprocessor=None, lowercase=False)),
                ('model', LatentDirichletAllocation(n_topics=self.n_topics)),
                ])
                
    def fit_transform(self, documents):
        self.model.fit_transform(documents)
        return self.model
    
    def get_topics(self,n=10):
        """
        n is the number of top terms to show for each topic
        """
        vectorizer = self.model.named_steps['vect']
        model = self.model.steps[-1][1]
        names = vectorizer.get_feature_names()
        topics = dict()
        
        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n-1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens
            
        return topics
    
if __name__ == '__main__':
    lda = SklearnTopicModels()
    documents = corpus
    lda.fit_transform(corpus)
    topics=lda.get_topics()
    #for topic, terms in topics.items():
        #print("Topic #{}:".format(topic+1))
        #print(terms)

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize

word = corpus.words()
corpus = [list(word_tokenize(doc)) for doc in corpus.raw()]
corpus = [
        TaggedDocument(words, ['d{}'.format(idx)])
        for idx, words in enumerate(corpus)
        ]
model = Doc2Vec(corpus, size=5, min_count=0)
print(model.doc2vecs[0])
