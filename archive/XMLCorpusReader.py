from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader import XMLCorpusReader
import codecs, os, bs4, nltk

cat_pattern = r'(a-z_\s]+)/.'
doc_pattern = r'[\w\d_]+\.xml'

class CIPSEACorpusReader(XMLCorpusReader, CorpusReader):
    def __init__(self, root, fileids=doc_pattern, encoding='latin=1', **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = cat_pattern
            
        XMLCorpusReader.__init__(self,fileids, kwargs)
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
            yield doc
            
    def paras(self, fileids=None, categories=None):
        for html in self.html(fileids, categories):
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(tags):
                yield element.text
            soup.decompose()

tags = ['h1','h2','h3','h4','h5','h6','h7','p','li','RSPNTOTH']        
corpus = CIPSEACorpusReader(")

for c in corpus:
    print(c)