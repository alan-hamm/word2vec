/* export medications from med-kaggle.csv to jsonl file */
/* author: alan hamm(pqn7)	*/
/* date: april 12 2022 		*/

libname m "\\cdc.gov\CSP_Private\M728\pqn7\poc_prodigy\utility_files";
filename t "\\cdc.gov\CSP_Private\M728\pqn7\poc_prodigy\utility_files\med_kaggle.txt";

/* read text file */
data m.med_kagg;
	if _n_=1 then do;
		regex=prxparse('s/([[:punct:]])//');
		retain regex;
	end;

	infile t truncover;
	input drugName $char512.;
	drugName=scan(drugname,1);
	if length(drugName) > 3;
	drugName=prxchange(regex,-1,drugName);

	keep drugName;
run;

/* write jsonl */

filename t "\\cdc.gov\CSP_Private\M728\pqn7\poc_prodigy\drug_pattern.jsonl";
data med_pattern;
	length lemma $ 512;
	set m.med_kagg;
	file t;

	*tmp= '{"label":"MEDICATION","pattern":"' || upcase(strip(drugName)) || '"}';
	lemma = '{"pattern":[{"lemma":{"regex":"(?i)' || strip(drugName) || '"}}], "label": "DRUG"}';
	put lemma;
run;


	
	
