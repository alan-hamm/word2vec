'''
program to parse f7notes to search for PII using spaCy NLP
original f7 notes datafile located: .../<month>/preview/level3/<month>_f7notes
text version of f7 notes created by SAS program

author: alan hamm(pqn7@cdc.gov)
date: september 2021
'''

from spacy import displacy
from spacy.attrs import ENT_TYPE
import spacy

nlp = spacy.load(r"\\cdc.gov\CSP_Private\M728\pqn7\poc_prodigy\span_spacy1\model-best")


#read F7NOTES dataset
from sas7bdat import SAS7BDAT
df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\Q1\level3\q1_f7note.sas7bdat").to_data_frame()
#df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\tmp_original.sas7bdat").to_data_frame()


#return True if ANY entity found
def find_entity(entity):
    ents=list(entity.ents)
    for i in ents:
        if i.label_.isalpha():
            return True
    
for index, row in df.iterrows():
    #clean F7note for better syntax and NER accuracy
    #see https://v2.spacy.io/models/en#en_core_web_lg
    tmp=row['F7NOTE'].replace('  ', ' ').replace(',','').strip()
    
    #create Doc object of F7note string
    doc=nlp(tmp)
    #if find_entity(doc):
    assert doc.has_annotation("ENT")
    if find_entity(doc):
        span=doc[1:]
        print(type(span))
