'''
program to produce dependency tree of question text

author: alan hamm(pqn7@cdc.gov)
date: october 2021
'''

from spacy import displacy

#initialize english spaCy model from from previously saved 
#'pip install' of en_core_web_lg-2.2.0.tar.gz having used 'model.to_disk' function
import spacy
nlp = spacy.load("\\\\cdc.gov\\project\\CCHIS_NCHS_DHIS\\HIS_ALL\\Databases\\NLP\\en")

#read question text dataset
from sas7bdat import SAS7BDAT
df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\question_text_qc\question_text2021NOV.sas7bdat").to_data_frame()

#process Question Text  
html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\question_text_qc\\question_text2021NOV.html", 'w')
for index, row in df.iterrows():
    tmp=row['QuestionText'].strip()
    
    #create Doc object of F7note string
    doc=nlp(tmp)
    
    #begin write html report
    svg = displacy.render(doc, style="dep", page=False, jupyter=False)
    html.write('<p style="background-color:LightGrey;">')
    html.write('<b><strong>')
    html.write(row['Name'])
    html.write('</b></strong></p>')
    html.write(svg)  
html.close()  
    