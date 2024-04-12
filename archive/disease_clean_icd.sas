				* 			author: alan hamm(pqn7)			*
			*					date: march 2022				*			  						
		*															*
	*															   		*
*																			*
* disease_clean_icd.sas														 *
* program to etl icd9 text dataset  		into text file					 *
*****************************************************************************;

libname o_mycsp "\\cdc.gov\CSP_Private\M728\pqn7\prodigy_poc\utility_files";
filename i "\\cdc.gov\CSP_Private\M728\pqn7\prodigy_poc\utility_files\icd9_from_cdc.txt";
filename o "\\cdc.gov\CSP_Private\M728\pqn7\prodigy_poc\diseases_cleaned.jsonl";


data o_mycsp.diseases;
	length l $ 512;
	infile i truncover;
	input d $char32767.;

	if _n_=1 then do;
		clean=prxparse('s/([[:^ascii:]])//');
		word=prxparse('/[A-Z]{5,}/i');
		retain clean word;
		call missing(l);
	end;

	d=prxchange(clean,-1,d);
	
	start=1;
	stop=length(strip(d));

	call prxnext(word, start, stop, d, pos, len);
	do while(pos > 0);
		l = substr(d,pos,len);
		output;
		call prxnext(word, start, stop, d, pos, len);
	end;

	keep l;
run;
