				* 			author: alan hamm(pqn7)			*
			*					date: march 2022				*			  						
		*															*
	*															   		*
*																			*
* prodigy_poc.sas															 *
* program to etl multiple f7notes datasets into text file					 *
*****************************************************************************;
options nospool; %global year;
%macro etl(year=2022);
	
	libname o_mycsp "\\cdc.gov\CSP_Private\M728\pqn7\poc_f7note_files";
	filename o "\\cdc.gov\CSP_Private\M728\pqn7\poc_f7note_files\private_ai.jsonl";

	%if %eval(&year=2022) %then %do;
	libname a "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\M01\Preview\Jan1-31\level3" access=readonly;
	libname b "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\M02\Preview\level3" access=readonly;
	libname c "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\M03\Preview\level3" access=readonly;
	libname d "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\M04\Preview\level3" access=readonly;
	%end;

	%if %eval(&year=2021) %then %do;
		%do i=1 %to 4;
			libname q&i "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&year\Q&i\level3" access=readonly;
		%end;
	%end;

	data /*o_mycsp.*/f7notes_&year;
		if _n_ = 1 then do;
			regex=prxparse('s/(([[:^ascii:]])|([\"])|[\\])//');
			retain regex;
		end;

		%if %eval(&year=2021) %then %do; set q1.q1_f7note  q2.q2_f7note q3.q3_f7note q4.q4_f7note; %end;
		%if %eval(&year=2022) %then %do; set a.m01_f7note b.m02_f7note c.m03_f7note d.m04_f7note; %end;

		if HHID in ("0806085UN22053004","0812105UN22012701","0813243AN22003001","0813243AN22003003","0813121UN22026302");

		if anyalnum(f7note)=0 then delete;
		
		f7note=prxchange(regex,-1,f7note);
		keep f7note;
	run;

	data _null_;
		set f7notes_&year end=last;
		file o lrecl=512;
		tmp=compbl(strip(f7note));
		put tmp;

		retain c 0;
		c+countw(f7note);
		if last then put "NOTE: number of wrds " c=;
	run;
%mend; %etl()
		
*proc surveyselect data=f7notes_2021 out=e2021 noprint outorder=random sampsize=100000 selectall method=srs; run;
*proc surveyselect data=f7notes_2022 out=e2022 noprint outorder=random sampsize=100000 selectall method=srs; run;
/*
data _null_;
		length tmp $ 5000;
		set o_mycsp.e2021;
		file o;
		
		tmp= '{"text": "' || strip(f7note) || '"}';
		tmp=strip(tmp);
		put tmp;
run;

data _null_;
		length tmp $ 5000;
		set o_mycsp.e2022;
		file o mod;
		
		tmp= '{"text": "' || strip(f7note) || '"}';
		tmp=strip(tmp);
		put tmp;
run;
libname _all_ clear;
*/

