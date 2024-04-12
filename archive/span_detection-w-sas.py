'''
program to parse f7notes to search for PII using spaCy NLP
original f7 notes datafile located: .../<month>/preview/level3/<month>_f7notes
text version of f7 notes created by SAS program

author: alan hamm(pqn7@cdc.gov)
date: september 2021
'''

from spacy import displacy

#initialize english spaCy model from 'pip install' of en_core_web_lg-2.2.0.tar.gz
#import en_core_web_lg
#nlp= en_core_web_lg.load()

#initialize english spaCy model from from previously saved 
#'pip install' of en_core_web_lg-2.2.0.tar.gz having used 'model.to_disk' function
import spacy
nlp = spacy.load("\\\\cdc.gov\\CSP_Private\\M728\\pqn7\\poc_prodigy\\span_out1\\model-best")


#read F7NOTES dataset
from sas7bdat import SAS7BDAT
df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\\2021\M11\Preview\Nov16\level3\m11_f7note.sas7bdat").to_data_frame()
#df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\tmp_original.sas7bdat").to_data_frame()


# return True if f7note contains PERSON entity
def find_person(entity):
    ents=list(entity.ents)
    for i in ents:
        if i.label_ == 'PERSON':
            return True

#return True if ANY entity found
def find_entity(entity):
    ents=list(entity.ents)
    for i in ents:
        if i.label_.isalpha():
            return True

#process F7 notes  
'''
********************
!!! USE FOR DEMO !!!
********************

EXTRACT ALL F7NOTES
LABEL ONLY F7NOTE strings WHERE person ENTITY FOUND
SET BACKRGROUND TO GREY WHERE NO ENTITY FOUND
'''
html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\2021\\QualityControl\\span_poc.html", 'w')
#html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\sas7bdat-original-cleaned-ANY-notes-WITH-person-entity.html", 'w')
for index, row in df.iterrows():
    #clean F7note for better syntax and NER accuracy
    #see https://v2.spacy.io/models/en#en_core_web_lg
    tmp=row['F7NOTE'].replace('  ', ' ').replace(',','').strip()
    
    #create Doc object of F7note string
    doc=nlp(tmp)
    
    #used in testing non-cleaned f7note strings
    #doc=nlp(row['F7NOTE'])
    
    #begin write html report
    html.write('<table>')
    html.write('<tr>')
    html.write('<th align="left">HHID</th>')
    html.write('<th align="left">F7Note</th>')
    html.write('</tr>')
    html.write('<b><strong><p style="font-size:105%;">')
    html.write('<tr><td>')
    html.write('<b><strong><p style="font-size:105%;">')
    html.write(row['HHID'])
    html.write(':')
    html.write('</strong></b></p> <p style="font-size:100%;></p>"')
    svg = displacy.render(doc, style="span", page=False, jupyter=False)
            
    html.write('<tr><td>')
    html.write(svg)  
html.close()  