'''
alan hamm(pqn7@cdc.gov)
date august 2021

Intro to NLP with spaCy (1): Detecting programming languages | Episode 1: Data exploration
https://www.youtube.com/watch?v=WnGPv6HnBok
'''

from spacy.tokens import Doc
from spacy import displacy
import pandas as pd
import en_core_web_lg
from pathlib import Path
import random

#load NLP model
nlp=en_core_web_lg.load(disable='ner')

#read F7note csv file
df=(pd.read_csv("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\training_exercises\\test_sentences.csv",
			encoding="ISO-8859-1", usecols=["text"]))			
f7notes = [_ for _ in df["text"]]

#read first_names csv file
df_names=(pd.read_csv("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\production_data\\first_names.csv",
			encoding="ISO-8859-1", usecols=["firstname"]))			
names_list = [_ for _ in df_names["firstname"]]

#randomize and select 20 sentences
tmp_f7=random.choices(f7notes, k=20)
#type(tmp_f7)
'''
#print randomly selected 20 items
for i in range(len(tmp_f7)):
	print(tmp_f7[i])
'''

#make string objects
tmp_f7_str=''.join([str(elem) for elem in f7notes]) 
tmp_name_str=''.join([str(elem) for elem in names_list]) 

#make NLP objects
doc_f7 = nlp(tmp_f7_str)
doc_names = nlp.pipe(tmp_name_str)

#write dependency tree visualization
svg = displacy.render(doc_f7, style="dep", page=True)
output_path = Path("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\training_exercises\\dependecy_graph.html")
output_path.open("w", encoding="utf-8").write(svg)

'''
#GARBAGE -- replacing <for t.text...> with <if t.text...>
def has_name(doc):
    for t in doc:
    	for t.text in nlp.pipe(tmp_name_str):
            return True
    return False	  	
#generator object to iterate where has_name = True	 
g = (doc_f7 for doc_f7 in nlp.pipe(f7notes) if has_name(doc_f7))
[next(g) for i in range(20)]
'''


#THIS WORKS
def has_name(doc):
    for t in doc:
    	if t.text in ["KASEY", "KHALID","LAKSHMI","RENATA","RYSZARD"]:
            return True
    return False
    
g = (doc_f7 for doc_f7 in nlp.pipe(f7notes) if has_name(doc_f7))
[next(g) for i in range(5)]