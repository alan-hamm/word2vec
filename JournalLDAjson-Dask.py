

#%%
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
import nltk.data
from nltk import sent_tokenize, pos_tag, wordpunct_tokenize
import en_core_web_lg
import gensim
from gensim.models import ldamulticore
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
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

import multiprocessing
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from time import time  # To time our operations

from sklearn.manifold import TSNE
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm

import pandas as pd
#import modin.pandas as pd

from tqdm import tqdm, tqdm_notebook
import csv


#%%
# we create a list to contain the json files that are to be processed

#year = 2019
#DOC_ID=list()
#for x in os.listdir(r"C:/_harvester/data/html-by-year/10s"):
#    if x.endswith(".json") and x[:4] in ['2019']:
#        DOC_ID.append(x)
#print(DOC_ID)
DOC_ID =r'.*([\d]+_html\.json)'


# we create a list of categories/keywords/tags to
#cat_pattern = r'(.*)[\d]_html\.json'
#cat_pattern = r'(.*?)(\d{,4}?_html\.json'
#CAT_PATTERN = r'(.*?)\d{,4}\.[\w]+'
CAT_PATTERN = r'^(.*?)[\W]*?\d{,4}?_html\.json'


# we mark the HTML tags to be used for 
# extacting the desired article, etc. text
# don't include 'li' tag e.g. <li>The Centers for Disease Control and Prevention (CDC) cannot attest to the accuracy of a non-federal website.</li>
TAGS = ['p']
#TAGS = ['h1']

# stop words
stop_words = stopwords.words('english')
# observed findings 
stop_words.extend(['icon', 'website', 'mmwr', 'citation', 'author', 'report', 'formatting', "format",'regarding',
                   'system', 'datum', 'link', 'linking', 'federal', 'data', 'tract', 'census', 'study',"question",
                   'conduct', 'report', 'including', 'top', 'summary', 'however', 'name', 'known', 'figure', 'return', 
                   'page', 'view', 'affiliation', 'pdf', 'law', 'version', 'list', 'endorsement', "review",
                   'article', 'download', 'reference', 'publication', 'discussion', 'table', 'vol', "message",
                   'information', 'web', 'notification', 'policy', 'policie', #spaCy lemmatization can make errors with pluralization(e.g. rabie for rabies)
                   'acknowledgment', 'altmetric',
                   'abbreviation', 'figure', "service","imply","current","source",
                   "trade","address", "addresses","program","organization" ,"provided", "copyrighted", "copyright",
                   "already", "topic", "art", 'e.g', 'eg'])

# pretrained model for POS tagging/filtering
nlp = en_core_web_lg.load( disable=['parser','ner'])

# set encoding for CorpusReader class
ENCODING = 'utf8'

# SET DIR PATHS
JSON_OUT = "C:/_harvester/data/json-outputs/"

# set the minimum number of topics to find
MIN_TOPICS = 100

# set the maximum number of topics to find
MAX_TOPICS = 505

# set the step by value
STEP_BY = 2

# set value to determine if lemmatization will be performed
LEMMATIZATION = True


#%%
import codecs
import json
import bs4
import re
import nltk
from time import time
import dask.distributed as dd
from dask.distributed import Client, LocalCluster #, LocalCUDACluster
from dask.diagnostics import ProgressBar
import dask

class JOURNALCorpusReader(CategorizedCorpusReader, CorpusReader):
    """ a corpus reader for CDC Journal articles """
    
    def __init__(self, root, tags=TAGS, fileids=DOC_ID, encoding=ENCODING, **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)
        
        self.tags = tags

    def resolve(self, fileids=None, categories=None):
        if categories is not None:
            return self.fileids(categories)
        
        return fileids

    def docs(self,fileids=None, categories=None):
        fileids = self.resolve(self.fileids(), self.categories())
        
        for path, encoding in self.abspaths(self.fileids(), include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield json.load(f)

    def html(self, fileids=None, categories=None):
        for idx, doc in enumerate(self.docs(fileids=fileids,categories=categories)):
            #pp.pprint(f"The file {self.fileids()[idx]} is being processed in HTML()")
            for sentence in doc:
                try:
                    yield Paper(sentence).summary()
                except Unparseable as e:
                    print("Could not parse HTML: {}".format(e))
                    print(f"the fileid {self.fileids()[idx]}")
                    pp.pprint(sentence)
                    print("\n")
                    continue
   
    def paras(self,fileids=None,categories=None):
        for html in self.html(fileids=fileids,categories=categories):
            soup=bs4.BeautifulSoup(html,'html.parser')
            for element in soup.find_all(TAGS):
                yield element.text
            soup.decompose()
              
    def sents(self,fileids=None,categories=None):
        for paragraph in self.paras(fileids=fileids,categories=categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence
                
    def words(self, fileids=None, categories=None):
        for paragraph in self.paras(fileids=fileids, categories=categories):
            for sentence in self.sents(fileids=fileids, categories=categories):
                for token in wordpunct_tokenize(sentence):
                    yield token

    def generate(self, fileids=None, categories=None):
        started = time()

        # Specify the local directory path
        DASK_DIR = '/_harvester/tmp-dask-out'

        # Deploy a Single-Machine Multi-GPU Cluster
        # https://medium.com/@aryan.gupta18/end-to-end-recommender-systems-with-merlin-part-1-89fabe2fa05b
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device IDs
        protocol = "tcp"  # "tcp" or "ucx"
        num_gpus = 1
        NUM_GPUS=[0]
        cores = multiprocessing.cpu_count() - 1 # Count the number of cores in a computer
        visible_devices = ",".join([str(n) for n in NUM_GPUS])  # Select devices to place workers
        device_limit_frac = 0.7  # Spill GPU-Worker memory to host at this limit.
        device_pool_frac = 0.8
        part_mem_frac = 0.15

        # Manually specify the total device memory size (in bytes)
        device_size = 10 * 1024 * 1024 * 1024  # GPU has 12GB but setting at 10GB
                
        ram_memory_limit = "75GB" # Set the RAM memory limit (per worker)
        device_limit = int(device_limit_frac * device_size)
        device_pool_size = int(device_pool_frac * device_size)
        part_size = int(part_mem_frac * device_size)

        cluster = LocalCluster(
                n_workers=(multiprocessing.cpu_count()-2),
                threads_per_worker=2,
                #processes=False,
                memory_limit=ram_memory_limit,
                local_directory=DASK_DIR,
                dashboard_address=":8787",
                protocol="tcp",
        )

        client = dd.Client(cluster)  # Connect to the local cluster

        # Get information about workers from scheduler
        workers_info = client.scheduler_info()["workers"]

        # Iterate over workers and set their memory limits
        for worker_id, worker_info in workers_info.items():
            worker_info["memory_limit"] = ram_memory_limit

        # Verify that memory limits have been set correctly
        #for worker_id, worker_info in workers_info.items():
        #    print(f"Worker {worker_id}: Memory Limit - {worker_info['memory_limit']}")

        # verify that Dask is being used in your code, you can check the following:
        # Check if the Dask client is connected to a scheduler:
        if client.status == "running":
            print("Dask client is connected to a scheduler.")
            # Scatter the embedding vectors across Dask workers
        else:
            print("Dask client is not connected to a scheduler.")

        # Check if Dask workers are running:
        if len(client.scheduler_info()["workers"]) > 0:
            print("Dask workers are running.")
        else:
            print("No Dask workers are running.")

        # Structures to perform counting
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()
        
        # Create Dask delayed objects for each method
        paras_list = list(self.paras(fileids=fileids, categories=categories))
        sents_list = list(self.sents(fileids=fileids, categories=categories))
        words_list = list(self.words(fileids=fileids, categories=categories))

        paras_delayed = dask.delayed(paras_list)
        sents_delayed = dask.delayed(sents_list)
        words_delayed = dask.delayed(words_list)

        # Enable the Dask progress bar
        ProgressBar().register()

        # Compute the delayed objects in parallel using Dask's distributed scheduler
        with ProgressBar():
            para_dict = dict(enumerate(paras_delayed.compute(), desc="Processing paragraphs"))
            sent_dict = dict(enumerate(sents_delayed.compute(), desc="Processing sentences"))
            word_dict = dict(enumerate(words_delayed.compute(), desc="Processing words"))

        # Compute the number of files
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())

        # Return data structure with information
        meta = {
            'files': self.fileids(),
            'nfiles': n_fileids,
            'paras': len(para_dict),
            'sents': len(sent_dict),
            'words': len(word_dict),
            'vocab': len(tokens),
            'lexdiv': float(len(word_dict)) / float(len(tokens)),
            'wdps': float(len(word_dict)) / float(len(sent_dict)),
            'sppar': float(len(sent_dict)) / float(len(para_dict)),
            'mins': round((time() - started) / 60, 2)
         }

        # Close connection to the Dask client and cluster
        client.close()
        cluster.close()

        return meta, para_dict, sent_dict, word_dict, counts, tokens



#%%
_corpus = JOURNALCorpusReader('/_harvester/data/html-by-year/10s')
#print(_corpus.categories())
_corpus.fileids()

#%%
corpus_tuple = _corpus.generate()

# %%
