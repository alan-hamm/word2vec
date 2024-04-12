# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from IPython.core.display import display, HTML
import spacy
import en_core_web_lg
from spacy import displacy
import io


nlp = en_core_web_lg.load()

ff = io.open('\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\m1_m2f7noteoutput.txt', 'r', encoding='utf-8')
doc_ent=nlp(ff.read())
ff.close()


'''html = displacy.render([doc_ent], style="ent", page=True)'''


svg = displacy.render(doc_dep, style="dep")

file = open('\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\m1_m2f7noteoutput.html',"w")

file.write(svg)
file.close()
