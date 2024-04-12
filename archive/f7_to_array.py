
#read F7NOTES dataset
from sas7bdat import SAS7BDAT
df=SAS7BDAT(r"\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\M08\level3\m08_f7note.sas7bdat").to_data_frame()


f7array=[]
for index, row in df.iterrows():
	tmp=row['F7NOTE'].replace('  ', ' ').replace(',','').strip()
	f7array.append(tmp)