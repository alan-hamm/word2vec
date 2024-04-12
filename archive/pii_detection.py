'''
program to parse f7notes to search for PII using spaCy NLP
original f7 notes datafile located: .../<month>/preview/level3/<month>_f7notes
text version of f7 notes created by SAS program

author: alan hamm(pqn7@cdc.gov)
date: september 2021
'''

from spacy import displacy
import pandas as pd

#initialize english spaCy model from 'pip install' of en_core_web_lg-2.2.0.tar.gz
#import en_core_web_lg
#nlp= en_core_web_lg.load()

#initialize english spaCy model from from previously saved 
#'pip install' of en_core_web_lg-2.2.0.tar.gz having used 'model.to_disk' function
import spacy
nlp = spacy.load("\\\\cdc.gov\\project\\CCHIS_NCHS_DHIS\\HIS_ALL\\Databases\\NLP\\en")

#read F7note csv file
df=(pd.read_csv("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\proof_of_concept.csv",
	encoding="ISO-8859-1", usecols=["HHID","F7NOTE"]))	


# boolean to determine if NER label is PERSON
def find_person(entity):
    ents=list(entity.ents)
    for i in ents:
        if i.label_ =='PERSON':
            return True

#return all f7notes independent of entity being PERSON
'''
def find_person(entity):
    ents=list(entity.ents)
    for i in ents:
        if i.label_ =='PERSON':
            return True
    return True
'''

#process F7 notes
html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\proof_of_concept.html", 'w')
for index, row in df.iterrows():
    #create Doc object of F7note string
    doc=nlp(row['F7NOTE'])
    if list(doc.ents) !=[] and find_person(doc):
        #render Doc.sents if entity.label_ = 'PERSON' is true
        svg = displacy.render(doc, style="ent", page=False, jupyter=False)
            
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
        html.write('<tr><td>')
        html.write(svg)
        html.write('</td></tr>')
        html.write('</table>')
html.close()  


#  RESULTS IN THE FOLLOWING AS NOT ALL NOTES CONTAIN ENTITIES THAT CAN BE VISUALIZED
#   UserWarning: [W006] No entities to visualize found in Doc object. If this is surprising to you, make sure 
#   the Doc was processed using a model that supports named entity recognition, and check the `doc.ents` property 
#   manually if necessary.
'''
html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\proof_of_concept.html", 'w')        
for index, row in df.iterrows():
    #create Doc object of F7note string
    doc=nlp(row['F7NOTE'])
    #render Doc.sents if entity.label_ = 'PERSON' is true
    svg = displacy.render(doc, style="ent", page=False, jupyter=False)
        
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
    html.write('<tr><td>')
    html.write('</strong></b></p> <p style="font-size:100%;></p>"')
    html.write(svg)
    html.write('</td></tr>')
    html.write('</table>')
html.close()
'''
     
'''
!!! THIS CODE WORKS !!!
html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\proof_of_concept.html", 'w')
for index, row in df.iterrows():
    doc=nlp(row['F7NOTE'])
    for ent in doc.ents:
        if find_person(ent):
            svg = displacy.render(doc, style="ent", page=False, jupyter=False)
            html_table_b='<table><tr><th>HHID</th><th></th></tr>'
            text_style_begin='<b><strong><p style="font-size:105%;">'
            text_style_end='</strong></b></p> <p style="font-size:100%;></p>"'
            hhid_style= text_style_begin + row['HHID']  + ':   ' + text_style_end
            tmp_str=html_table_b + text_style_begin + '<tr><td>' + hhid_style + '</td><td>' + text_style_end + svg + '</td></tr></table>'
            html.write(tmp_str)
html.close()  
'''

'''
!!!!THIS CODE WORKS!!!!!
html= open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\proof_of_concept.html", 'w')
for index, row in df.iterrows():
    doc=nlp(row['F7NOTE'])
    for ent in doc.ents:
        if find_person(ent):
            svg = displacy.render(doc, style="ent", page=False, jupyter=False)
            html_table_b='<table><tr><th>HHID</th><th>    </th></tr>'
            text_style_begin='<b><strong><p style="font-size:105%;">'
            text_style_end='</strong></b></p> <p style="font-size:100%;></p>"'
            tmp_str=html_table_b + text_style_begin + '<tr><td>' + row['HHID'] + ':' +  '</td><td>' + text_style_end + svg + '</td></tr></table>'
            #tmp_str='HHID: ' + row['HHID'] + ' || PERSON ' + str(doc.ents) + '<br> || F7note: ' + svg
            html.write(tmp_str)
html.close()          
'''         

    
    
