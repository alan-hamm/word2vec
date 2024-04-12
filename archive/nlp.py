import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import json


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return(data)
    
hp_chars=load_data("\\\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\SpiderCorpora\Apr2021_qt.json")
print(hp_chars)

nlp = spacy.load("\\\\cdc.gov\\project\\CCHIS_NCHS_DHIS\\HIS_ALL\\Databases\\SpiderCorpora\\en_core_web_lg-2.3.1\\en_core_web_lg")
doc = nlp(sentences[0])
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)