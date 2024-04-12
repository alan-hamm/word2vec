from spacy import displacy
import pandas as pd
import spacy
import en_core_web_lg

nlp=en_core_web_lg.load()

#read F7note csv file
df=(pd.read_csv("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\proof_of_concept.csv",
			encoding="ISO-8859-1", usecols=["F7NOTE"]))			

file = open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\proof_of_concept.html","w")
for i in df["F7NOTE"]:
	f7lines=''.join(i)
	doc=nlp(f7lines)
	html = displacy.render([doc], style="ent")
	file.write(html)
	
file.close()