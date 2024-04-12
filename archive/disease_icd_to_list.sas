				* 			author: alan hamm(pqn7)			*
			*					date: march 2022				*			  						
		*															*
	*															   		*
*																			*
* icd9_to_list.sas															 *
* program to etl icd9 text dataset  		into text file					 *
*****************************************************************************;

libname o_mycsp "\\cdc.gov\CSP_Private\M728\pqn7\prodigy_poc\utility_files";
filename i "\\cdc.gov\CSP_Private\M728\pqn7\prodigy_poc\utility_files\icd9_from_cdc.txt";
filename o "\\cdc.gov\CSP_Private\M728\pqn7\prodigy_poc\diseases.jsonl";
filename m "\\cdc.gov\CSP_Private\M728\pqn7\prodigy_poc\diseases.txt";

data o_mycsp.diseases;
	length l $ 512;
	infile i truncover;
	input d $char32767.;

	if _n_=1 then do;
		clean=prxparse('s/\b([[:^ascii:]])\b//');
		word=prxparse('/[A-Z]{5,}/i');
		retain clean word;
		call missing(l);
	end;

	d=prxchange(clean,-1,d);
	
	start=1;
	stop=length(strip(d));

	call prxnext(word, start, stop, d, pos, len);
	do while(pos > 0);
		l = lowcase(substr(d,pos,len));
		output;
		call prxnext(word, start, stop, d, pos, len);
	end;

	keep l;
run;

proc sort data=o_mycsp.diseases noduprecs; by l; run;

data o_mycsp.diseases_jsonl;
	set o_mycsp.diseases;
	length tmp $ 1024;
	file o;
	tmp='{"label":"DISEASE","pattern":"' || strip(l) || '"}';
	put tmp;
run;

data _null_;
	set o_mycsp.diseases_jsonl;
	file o;
	put tmp;
run;

data _null_;
	file m;
	set o_mycsp.diseases;
	put l;
run;




