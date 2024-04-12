# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import en_core_web_lg
import spacy
from spacy import displacy
from pathlib import Path

nlp=en_core_web_lg.load()

doc = nlp("My name is Alan Hamm and I was born on 24th August 1984. \
           I work at Rasa from Haarlem. I just bought a guitar \
           cost $1000 on ebay and I will get is services here for 20 euro a year.")

svg = displacy.render(doc, style="ent", page=True)

output_path = Path("\\\\cdc.gov\\project\\CCHIS_NCHS_DHIS\\HIS_ALL\\Individual Folders\\Alan\\Code\\Reference\\NLP\\spaCy\\exercises\\exercise2.html")
output_path.open("w", encoding="utf-8").write(svg)

[(e, type(e)) for e in doc.ents]