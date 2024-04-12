from spacy.gold import docs_to_json
import pandas as pd
import en_core_web_lg
import srsly

nlp=en_core_web_lg.load()

#read F7note csv file
df=(pd.read_csv("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\evaluate.csv",
			encoding="ISO-8859-1", usecols=["text"]))			
f7notes = [_ for _ in df["text"]]

#make string objects
tmp_f7_str='\n'.join([str(elem) for elem in f7notes])

#create nlp object
doc=nlp(tmp_f7_str)

#gold parse
json_data= docs_to_json([doc])


srsly.write_json("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\jsonl_data.jsonl", json_data)