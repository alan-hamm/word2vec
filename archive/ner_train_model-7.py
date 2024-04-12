from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import pandas as pd

ruler=EntityRuler(nlp)
nlp.add_pipe(ruler)


#read F7note csv file
df=(pd.read_csv("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\evaluate.csv",
			encoding="ISO-8859-1", usecols=["text"]))			
f7notes = [_ for _ in df["text"]]

#make string objects
tmp_f7_str=' '.join([str(elem) for elem in f7notes])

#create nlp object
doc=nlp(tmp_f7_str)

print([(ent.text, ent.label_) for ent in doc.ents])