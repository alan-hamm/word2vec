libname a "\\cdc.gov\CSP_Private\M728\pqn7\poc_prodigy\utility_files";
filename i "\\cdc.gov\CSP_Private\M728\pqn7\poc_prodigy\utility_files\disease_pattern_icd_output.jsonl";
filename o "\\cdc.gov\CSP_Private\M728\pqn7\poc_prodigy\disease_pattern.jsonl";

data a.disease_pattern_jsonl;
	length pattern $ 1024 disease_l disease_u lemma basic $ 512;

	if _n_=1 then do;
	regex=('s/[[:punct:]]+/ /');
	retain regex;
	end;

	infile i truncover;	
	input pattern $char1024.;


	pattern=prxchange(regex,-1,pattern);

	disease_u = upcase(scan(pattern,4));

	basic='{"label":"DISEASE","pattern":"' || strip(disease_u) || '"}';
	
	disease_l=lowcase(scan(pattern,4));
	lemma = '{"pattern":[{"lemma":{"regex":"(?i)' || strip(disease_l) || '"}}], "label": "DISEASE"}';
run;

data _null_;
	set a.disease_pattern_jsonl end=last;
	file o;
	put lemma;
run;
