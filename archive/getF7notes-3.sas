/* getF7notes.sas
   
   Description: ELT F7 notes from Preview level 3
   Author: Alan Hamm(pqn7@cdc.gov)
   Date: June 2021
*/

* modify as needed;
%let year=2021;
%let month=m08;


*************************** DO NOT EDIT BELOW *************************************;
libname o "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&year\&month\Preview\level3" access=readonly;
filename p "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\proof_of_concept.txt";

data tmp_original(drop=reg);
	set o.&month._f7note(keep=hhid f7note);
	file p;

	if _n_=1 then do;
		reg=prxparse('s/[[:^ascii:]]//');
		retain reg;
	end;

	if anyalpha(f7note)=0 then delete;

	/* 9.13.2021 REMOVING NON-ASCII(keep 0-177) DECREASES SPACY ACCURACY */
	* remove non-ASCII characters;
	*call prxchange(reg,-1,f7note);
	
	* remove double quotation and comma;
	*f7note=translate(f7note,trimn(''),'"');
	*f7note=translate(f7note,trimn(''),',');
	*f7note=compbl(f7note);
	* write to txt file;
	put "HHID: " HHID " F7: " f7note;
run;
libname m "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\";
proc copy in=work out=m; select tmp_original; run;
proc export data=tmp
	 outfile="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\proof_of_concept-tmp_original.csv"
	 dbms=csv
	 replace;
run;
