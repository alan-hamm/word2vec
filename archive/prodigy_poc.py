'''
program to parse f7notes to search for PII using spaCy NLP
original f7 notes datafile located: .../<month>/preview/level3/<month>_f7notes
text version of f7 notes created by SAS program

author: alan hamm(pqn7@cdc.gov)
date: september 2021
'''

import io
from spacy import displacy
import spacy

nlp = spacy.load(r"\\cdc.gov\CSP_Private\M728\pqn7\poc_prodigy\train_out4\model-best")


#read F7NOTES dataset
from sas7bdat import SAS7BDAT
df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\Q1\level3\q1_f7note.sas7bdat").to_data_frame()
#df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\tmp_original.sas7bdat").to_data_frame()


#return True if ANY entity found
def find_entity(entity):
    ents=list(entity.ents)
    #subtree = [t.text for t in entity[0:entity.__len__].subtree ]
    for i in ents:
        if i.label_.isalpha():
            return True

#process F7 notes  
'''
EXTRACT ALL F7NOTES
LABEL any ENTITY FOUND IN F7NOTE string
SET BACKRGROUND TO GREY WHERE NO ENTITY FOUND
'''

html= open("\\\\cdc.gov\\CSP_Private\\M728\\pqn7\\poc_f7note_files\\train4.html", 'w')

for index, row in df.iterrows():
    #clean F7note for better syntax and NER accuracy
    #see https://v2.spacy.io/models/en#en_core_web_lg
    tmp=row['F7NOTE'].replace('  ', ' ').replace(',','').strip()
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
    if find_entity(doc):
        svg = displacy.render(doc, style="ent", page=False, jupyter=False)        
        html.write('<tr><td>')
        html.write(svg)
    else:
        html.write('</strong></b></p> <p style="font-size:100%;></p>"')
        html.write('<tr><td style="background-color:white;">')
        html.write(tmp)  
        html.write('</td></tr>')
    html.write('</table>')
html.close()  