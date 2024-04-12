disease=open("\\\\cdc.gov\\CSP_Private\\M728\\pqn7\\prodigy_poc\\diseases.txt",'w')
icd=open("\\\\cdc.gov\\CSP_Private\\M728\\pqn7\\prodigy_poc\\utility_files\\icd9_from_cdc.txt",'r')
lines=icd.readlines()
for line in lines:
	print(line)
	#word regular expression
	w=re.findall(r'\w+',line)
	disease.write(w)
	
disease.close()
icd.close()