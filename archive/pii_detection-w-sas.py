'''
program to parse f7notes to search for PII using spaCy NLP
original f7 notes datafile located: .../<month>/preview/level3/<month>_f7notes
text version of f7 notes created by SAS program

author: alan hamm(pqn7@cdc.gov)
date: september 2021
'''

from spacy import displacy
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL

config = {
        "threshold": 0.5,
        "model": DEFAULT_MULTI_TEXTCAT_MODEL,}

nlp.add_pipe("textcat_multilabel", config = config)

#initialize english spaCy model from 'pip install' of en_core_web_lg-2.2.0.tar.gz
#import en_core_web_lg
#nlp= en_core_web_lg.load()

#initialize english spaCy model from from previously saved 
#'pip install' of en_core_web_lg-2.2.0.tar.gz having used 'model.to_disk' function
import spacy
nlp = spacy.load(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\models\model-best")
#nlp = spacy.load(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\models\model-best")


#read F7NOTES dataset
from sas7bdat import SAS7BDAT
df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_MASTER\PII_Archive\2018\Q1\Level2\q1_visits_chi_confidential.sas7bdat").to_data_frame()
#df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\tmp_original.sas7bdat").to_data_frame()
df=df[['RSPNTOTH']].dropna()

# return True if f7note contains PERSON entity
def find_person(entity):
    ents=list(entity.ents)
    for i in ents:
        if i.label_ == 'PERSON':
            return True

#return True if ANY entity found
def find_entity(entity):
    ents=list(entity.labels)
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
'''
html= open(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\QualityControl\2020M04_F7_PERSONner.html", 'w')
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
    if find_person(doc):
        svg = displacy.render(doc, style="ent", page=False, jupyter=False)
            
        html.write('<tr><td>')
        html.write(svg)  
    else:
        html.write('</strong></b></p> <p style="font-size:100%;></p>"')
        html.write('<tr><td style="background-color:LightGrey;">')
        html.write(tmp)  
    html.write('</td></tr>')
    html.write('</table>')
html.close()  
'''    

'''
EXTRACT ALL F7NOTES
LABEL any ENTITY FOUND IN F7NOTE string
SET BACKRGROUND TO GREY WHERE NO ENTITY FOUND
'''

#html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\2021\\QualityControl\\M11_nov16-ALL-entity-label.html", 'w')
'''
for index, row in df.iterrows():
    #clean F7note for better syntax and NER accuracy
    #see https://v2.spacy.io/models/en#en_core_web_lg
    tmp=row['RSPNTOTH']
    if row['RSPNTOTH'] != ' ':
        #create Doc object of F7note string
        doc=nlp(tmp)
        
        #used in testing non-cleaned f7note strings
        #doc=nlp(row['F7NOTE'])
        
        #begin write html report
        html.write('<table>')
        html.write('<tr>')
        html.write('<th align="left">CTRLNUM</th>')
        html.write('<th align="left">RSPNTOTH</th>')
        html.write('</tr>')
        html.write('<b><strong><p style="font-size:105%;">')
        html.write('<tr><td>')
        html.write('<b><strong><p style="font-size:105%;">')
        html.write(row['CTRLNUM'])
        html.write(':')
        html.write('</strong></b></p> <p style="font-size:100%;></p>"')
        if find_entity(doc):
            svg = displacy.render(doc, style="ent", page=False, jupyter=False)
                
            html.write('<tr><td>')
            html.write(svg)  
        #else:
        #    html.write('</strong></b></p> <p style="font-size:100%;></p>"')
        #    html.write('<tr><td style="background-color:LightGrey;">')
        #    html.write(tmp)  
        html.write('</td></tr>')
        html.write('</table>')
html.close()  
'''

'''
!!! THIS CODE WORKS !!!

EXTRACT ALL F7NOTES
LABEL ONLY person IN F7STRINGS any ENTITY FOUND WITHIN string
SET BACKRGROUND TO GREY WHERE NO ENTITY FOUND
'''
'''
#options for displacy rendering
options = {"ents": ["PERSON"]}

html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\sas7bdat-original-cleaned-ONLY-person-label.html", 'w')
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
    if find_entity(doc):
        svg = displacy.render(doc, style="ent", page=False, jupyter=False, options=options)
            
        html.write('<tr><td>')
        html.write(svg)  
    else:
        html.write('</strong></b></p> <p style="font-size:100%;></p>"')
        html.write('<tr><td style="background-color:LightGrey;">')
        html.write(tmp)  
    html.write('</td></tr>')
    html.write('</table>')
html.close()  
'''


#!!! THIS CODE WORKS !!!
#html= open(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\reports\test.html", 'w', encoding='utf-8')
for index, row in df.iterrows():
    #clean F7note for better syntax and NER accuracy
    #see https://v2.spacy.io/models/en#en_core_web_lg
    tmp=row['RSPNTOTH'].replace('  ', ' ').replace(',','').strip()
    
    #create Doc object of F7note string
    doc=nlp(tmp)
    
    if doc.cats['GOODREC'] == 1:
        print('Hello, Friend')
    elif doc.cats['BADREC'] == 1:
        print("Goodbye, Friend")
    #svg = displacy.render(doc, style="ent", page=False, jupyter=False)
            
    #html.write(svg)  

#html.close()  


    
    
