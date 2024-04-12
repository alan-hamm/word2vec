# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from IPython.core.display import display, HTML
import spacy
from spacy import displacy
import io

nlp=spacy.load("en_core_web_sm")

f7_mod = spacy.load("\\\\cdc.gov\\project\\CCHIS_NCHS_DHIS\\HIS_ALL\\NLP\\model\\model-final")

ff = io.open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\2021\\QualityControl\\M06_JUN1-26\\F7NoteOutput.txt", 'r', encoding='utf-8')
doc_ent=f7_mod(ff.read())
ff.close()

d = f7_mod(list_of_test_texts[0])
d.cats

html = displacy.render([doc_ent], style="ent", page=True)


"""svg = displacy.render(doc_dep, style="dep")"""

file = open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\2021\\QualityControl\\M06_JUN1-26\\PIIner_Jun1-26.html","w")

file.write(html)
file.close()