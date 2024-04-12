libname z "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data";
filename x "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\evaulate.txt";

data _e;
	infile x truncover lrecl=256;
	input std $ char256.;
run;

proc surveyselect data=_e out=evaluate noprint outorder=random sampsize=89 selectall method=srs; run;

filename _out "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\evaluate_shredded.txt";
data evaluate;
	*retain ID 0;
	length text $ 256;

	file _out;
	set evaluate(keep=std rename=(std=text));

	if _n_=1 then do;
		regASCII=prxparse('s/[[:^ascii:]]//');
		regLINE=prxparse('s/(([\"])|([\,])|([\r]*))//');
		retain regASCII regLINE;
	end;

	* remove non-ASCII characters;
	call prxchange(regASCII,-1,text);

	* remove new-line and return characters;
	call prxchange(regLINE,-1,text);


	* write to txt file;
	text=strip(text) || '.';
	put text;
	
	* write to dataset for proc export csv;
	*text=text||'\n';
	output;
run;

proc export data=evaluate(keep=text)
		    outfile="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\evaluate.csv"
			dbms=csv
			replace;
run;
