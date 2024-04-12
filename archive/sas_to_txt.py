#read F7NOTES dataset

import re
from sas7bdat import SAS7BDAT
df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\Q1\level3\q1_f7note.sas7bdat").to_data_frame()

#f7notes= open("\\\\cdc.gov\\CSP_Private\\M728\\pqn7\\prodigy_poc\\text_category.txt", 'w')
f7notes= open(r"\\cdc.gov\CSP_Private\M728\pqn7\poc_f7note_files\mdl6_correct\f7noteq12022", 'w')
for index, row in df.iterrows():
    tmp=str(row['F7NOTE'].replace('  ', ' ').replace(',','').strip())
    
    #write f7notes to txt file
    new_line='\n'
    tmp += new_line
    f7notes.write(tmp)
    
    #write individual words from sentence
    #w=re.findall(r'\w+',tmp)
    #for t in w:
    #	#print(type(t))
    #	t += new_line
    #	f7notes.write(t)
f7notes.close()
