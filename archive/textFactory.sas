/* 	author: alan hamm
	date: October 2022

	description: program to perform data/text profiling of PII_MASTER
				 PERSONVISITCONFID, VISITS_CHI_CONFIDENTIAL, quarterly production level2 chi datasets
				 for variables CTOTHER PSPECLANG PNONCONOTH PSTRATOTH RSPNTOTH PRSPNDOTHSPECNOATTEMPT NCTPEROT NCTTELOT

*/

/* !!!!!!!!!!!!!!!! ASSESS WHY VARIABLES WITH ANY NUMBER OF MISSING VALUES ARE INCLUDED IN ADAM_<...> DATASETS. 
	!!!!!!!  	IT HAS SOMETHING TO DO WITH MATCH-MERGE or INTERLEAVE !!!!!!!!   */
/* !!!!!!!!!!!!!!!! INVESTIGATE PROC MI and PROC GLMSELECT		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!	*********************/



%let START_YR=2019;
%let END_YR = 2022;
%let START_QTR=1;
%let END_QTR=4;



******************!!!!!!!!!! DO NOT EDIT BELOW THIS LINE !!!!!!!!!*********************;
%* datasets created below for text processing. this macvar is used to count missingness per variable;
%let ds=adam_ctother adam_nctperot adam_ncttelot adam_pnonconoth adam_prspndoth
		adam_pspeclang adam_pstratoth adam_rspntoth /*adam_speclang*/ adam_specnoattempt adam_stratoth;

%let utility = \\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis;

%let rc=%sysfunc(grdsvc_enable(_ALL_, resource=CSP_MOD));
signon profile;
%syslput _user_;
rsubmit connectwait=yes;


* macro to count the number of missing values per variable;
%macro count_missing(param=);
	%do i = 1 %to %sysfunc(countw(&param, ' '));
		%let holder = %scan(&param,&i);

		proc transpose data=utility.&holder out=utility.test; var _character_; run;

		options nonotes;
		data utility.cnt_missing_row_char(keep=_name_ cnt_missing_character rename=(_name_=name cnt_missing_character=count_missing));
			label _name_ = ' ';
			set utility.test(keep=_character_);
			cnt_missing_character = cmiss( of _character_ );
		run;
		options notes;
		proc sql; drop table utility.test; quit;

		proc transpose data=utility.&holder out=utility.test; run;
		options nonotes;
		data utility.cnt_missing_row_num(keep= _name_ cnt_missing_numeric rename=(_name_=name cnt_missing_numeric=count_missing));
			label _name_ = ' ';
			set utility.test;
			cnt_missing_numeric = nmiss( of _numeric_ );
		run;
		options notes;
		proc sql; drop table utility.test; quit;

		data utility.%substr(&holder,6)_missing;
			set utility.cnt_missing_row_char utility.cnt_missing_row_num;
		run;

		proc sql;
			drop table utility.cnt_missing_row_char;
			drop table  utility.cnt_missing_row_num;
		quit;
	%end;
%mend count_missing;


%macro TextFactory(START_YEAR=&START_YR, END_YEAR=&END_YR, START_QUARTER=&START_QTR, END_QUARTER=&END_QTR);
	%global pervisits_vars chi_vars;

	%* verify that the start year is less than the end year;
	%if %eval(&START_YEAR > &END_YEAR) %then %do;
		%put USER: The start year %bquote('&START_YEAR') is greater than the End Year %bquote('&END_YEAR');
		%put ERROR: The program is aborting;
		%abort cancel;
	%end;

	%* collate datasets;
	%do year=&START_YEAR %to &END_YEAR;
		%* defensive programming -- delete dataset(s) if exists;
		%if %sysfunc(exist(utility.all_&year.persvisits)) 
			or %sysfunc(exist(utility.all_&year.cipsea_chi)) 
			or %sysfunc(exist(utility.all_&year.cipsea_chi)) %then %do;
			proc sql;	
				drop table utility.all_&year.persvisits;
				drop table utility.all_&year.cipsea_chi;
				drop table utility.all_&year.prod_chi;
			quit;
		%end;

		%do quarter=&START_QUARTER %to &END_QUARTER;
			libname master "\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_MASTER\PII_Archive\&year\Q&quarter\Level2" access=readonly;
			libname PRODchi "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&year\Q&quarter\level2\Contact History" access=readonly;
			libname utility "&utility";

			%* verify source datasets exist. these datasets are used for creting a blank table for subsequent use;
			proc sql;
				create table utility.all_&year.persvisits like master.q&quarter._persvisitsconfid(drop=frcode);
				alter table utility.all_&year.persvisits add year num, month num, quarter num, hour num, minute num, second num; 

				create table utility.all_&year.cipsea_chi like master.q&quarter._visits_chi_confidential(drop = intnumber lno ver);
				alter table utility.all_&year.cipsea_chi add year num, month num, quarter num, hour num, minute num, second num; 

				create table utility.all_&year.prod_chi like PRODchi.q&quarter._visits_chi;
				alter table utility.all_&year.prod_chi add year num, month num, quarter num, hour num, minute num, second num; 
			quit;

			%* prep persvisits by quarter;
			data utility.persvisits_q&quarter.&year; 
				set master.q&quarter._persvisitsconfid(drop=frcode); 
			run;

			%* prep PII_MASTER contact history instrument;
			data utility.CIPSEAchi_q&quarter.&year; 
				set master.q&quarter._visits_chi_confidential(drop=intnumber lno ver); 
			run;

			%* prep PRODUCTION contact history instrument;
			data utility.PRODchi_q&quarter.&year; 
				set PRODchi.q&quarter._visits_chi
					(drop=frdate frmnth fryear frhour frmin frtime frsec
						  cttype casecontact langlist no_noi no_noi_sum intv_qrt intv_mon ver frcode
						  clanguage_1 -- complete_thisint nctper_1 -- ncttel_99 noninter_1 -- people_incomplete
						  restart -- totalcount);
			run;

			%* NOTE: Verify these derived values against source datetime variables for difference;
			%* 		 If there is a difference, determine why;

			%* extract year, hour, minute, second from VISITNAME variable;
			data utility.persvisits_q&quarter.&year;
				length year month quarter hour minute second 8;
				set utility.persvisits_q&quarter.&year;

				if prxmatch("(&year)",visitdate) then year=&year;

				month=input(substr(visitdate,1,2),2.);
				if month in (1,2,3) then quarter=1;
				else if month in (4,5,6) then quarter=2;
				else if month in (7,8,9) then quarter=3;
				else if month in (10,11,12) then quarter=4;

				hour=input(substr(visitdate,9,2),2.);
				minute=input(substr(visitdate,11,2),2.);
				second=input(substr(visitdate,13),2.);
			run;

			%* extract year, hour, minute, second from VISITNAME variable;
			data utility.CIPSEAchi_q&quarter.&year;
				length year month quarter hour minute second 8;
				set utility.CIPSEAchi_q&quarter.&year;

				if prxmatch("(&year)",visitdate) then year=&year;

				month=input(substr(visitdate,1,2),3.);
				if month in (1,2,3) then quarter=1;
				if month in (4,5,6) then quarter=2;
				if month in (7,8,9) then quarter=3;
				if month in (10,11,12) then quarter=4;

				hour=input(substr(visitdate,9,2),2.);
				minute=input(substr(visitdate,11,2),2.);
				second=input(substr(visitdate,13),2.);
			run;

			%* extract year, hour, minute, second from VISITNAME variable;
			data utility.PRODchi_q&quarter.&year;
				length year month quarter hour minute second 8;
				set PRODchi.q&quarter._visits_chi;

				if prxmatch("(&year)",visitdate) then year=&year;

				month=input(substr(visitdate,1,2),3.);
				if month in (1,2,3) then quarter=1;
				if month in (4,5,6) then quarter=2;
				if month in (7,8,9) then quarter=3;
				if month in (10,11,12) then quarter=4;

				hour=input(substr(visitdate,9,2),2.);
				minute=input(substr(visitdate,11,2),2.);
				second=input(substr(visitdate,13),2.);
			run;

			%* merge PERSONVISITS;
			proc sort data=utility.all_&year.persvisits; by ctrlnum; run;
			proc sort data=utility.persvisits_q&quarter.&year; by ctrlnum; run;
			data utility.all_&year.persvisits;
				merge utility.all_&year.persvisits utility.persvisits_q&quarter.&year;
				by ctrlnum;
			run;

			%* merge PII_MASTER and PRODUCTION contact history instrument;
			proc sort data=utility.CIPSEAchi_q&quarter.&year; by ctrlnum; run;
			proc sort data=utility.PRODchi_q&quarter.&year; by ctrlnum; run;
			data utility.chi_q&quarter.&year;
				merge utility.CIPSEAchi_q&quarter.&year utility.PRODchi_q&quarter.&year;
				by ctrlnum;
			run; 

			%* merge PII_MASTER, PRODUCTION to all contact history instrument;
			proc sort data=utility.all_&year.cipsea_chi; by ctrlnum; run;
			proc sort data=utility.chi_q&quarter.&year; by ctrlnum; run;
			data utility.all_&year.chi; 
				merge utility.all_&year.cipsea_chi utility.chi_q&quarter.&year; 
				by ctrlnum;
			run;

			%* merge CHI to all_<year>PROD_chi dataset;
			proc sort data=utility.all_&year.prod_chi; by ctrlnum; run;
			proc sort data=utility.chi_q&quarter.&year; by ctrlnum; run;
			data utility.all_&year.prod_chi;
				merge utility.all_&year.prod_chi utility.chi_q&quarter.&year;
				by ctrlnum;
			run;
		%end;
	%end;

	%* get variable names per dataset;
	proc sql noprint;
		select name into :pervisits_vars separated by ' '
		from dictionary.columns
		where lowcase(libname)='utility' and lowcase(memname)='persvisits';
		%put NOTE: The Persion Visit Dataset Variables.;
		%put NOTE: &pervisits_vars;

		select name into :chi_vars separated by ' '
		from dictionary.columns
		where lowcase(libname)='utility' and lowcase(memname)='chi';
		%put NOTE: The CHI Dataset Variables;
		%put NOTE: &chi_vars;
	quit;
		
	%* combine all yearly, quarterly datasets into a single dataset;
	data utility.all_persvisits(compress=yes);
		set 
		%do year=&START_YEAR %to &END_YEAR;
			utility.all_&year.persvisits
		%end;
		;
	run;

	%* combine all contact history instrument datasets into a single dataset;
	data utility.all_PRODchi(compress=yes);
		set 
		%do year=&START_YEAR %to &END_YEAR;
			utility.all_&year.prod_chi
		%end;
		;
	run;

	%* combine all CIPSEAS contact history instrument datasets into a single dataset;
	data utility.all_CIPSEAchi(compress=yes);
		set 
		%do year=&START_YEAR %to &END_YEAR;
			utility.all_&year.chi
		%end;
		;
	run;

	* sort for merging;
	proc sort data=utility.all_persvisits(compress=yes); by CTRLNUM; run;
	proc sort data=utility.all_PRODchi(compress=yes); by CTRLNUM; run;
	proc sort data=utility.all_CIPSEAchi(compress=yes); by CTRLNUM; run;

	* match-merge outcome, all_persvisits, and all_chi;
	libname utility "&utility";
	data utility.ADaM(compress=yes );
		merge utility.all_persvisits(in=a) utility.all_PRODchi(in=b) utility.all_CIPSEAchi(in=c);
		by ctrlnum;
		if a and b and c;
	run;

	*****************************************************;
	* create individual text analysis data model datasets;
	*****************************************************;
 
	%* PROCESS Analysis Data Model: CTOTHER;
	data utility.ADaM_CTOTHER(compress=yes);
		set utility.ADaM(drop=CTTYPE CASECONTACT PNONCONOTH PSPECLANG PSTRATOTH PRSPNDOTH P_COUNT SPECNOATTEMPT NCTPEROT NCTTELOT STRATOTH RSPNTOTH FRDATE -- FRSEC
									  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(ctother) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_CTOTHER nodupkeys; by CTRLNUM FINALOUTCOME CTOTHER; run;

	%* PROCESS Analysis Data Model: PNONCONOTH;
	data utility.ADaM_PNONCONOTH(compress=yes);
		set utility.ADaM(drop=CTOTHER CTTYPE CASECONTACT PSPECLANG PSTRATOTH PRSPNDOTH P_COUNT SPECNOATTEMPT NCTPEROT NCTTELOT SPECLANG STRATOTH RSPNTOTH FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER);
		where not missing(PNONCONOTH) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_PNONCONOTH nodupkeys; by CTRLNUM FINALOUTCOME PNONCONOTH; run;

	%* PROCESS Analysis Data Model: PSPECLANG;
	data utility.ADaM_PSPECLANG(compress=yes);
		set utility.ADaM(drop=CTOTHER CTTYPE CASECONTACT PNONCONOTH PSTRATOTH PRSPNDOTH P_COUNT SPECNOATTEMPT NCTPEROT NCTTELOT SPECLANG STRATOTH RSPNTOTH  FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(PSPECLANG) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_PSPECLANG nodupkeys; by CTRLNUM finaloutcome PSPECLANG; run;

	%* PROCESS Analysis Data Model: PSTRATOTH;
	data utility.ADaM_PSTRATOTH(compress=yes);
		set utility.ADaM(drop=CTOTHER PNONCONOTH PSPECLANG PRSPNDOTH P_COUNT SPECNOATTEMPT NCTPEROT NCTTELOT SPECLANG STRATOTH RSPNTOTH  FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(PSTRATOTH) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_PSTRATOTH nodupkeys; by CTRLNUM finaloutcome PSTRATOTH; run;

	%* PROCESS Analysis Data Model: PRSPNDOTH;
	data utility.ADaM_PRSPNDOTH(compress=yes );
		set utility.ADaM(drop=CTOTHER PNONCONOTH PSPECLANG PSTRATOTH P_COUNT SPECNOATTEMPT NCTPEROT NCTTELOT SPECLANG STRATOTH RSPNTOTH  FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(PRSPNDOTH) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_PRSPNDOTH nodupkeys; by CTRLNUM finaloutcome PRSPNDOTH; run;

	%* PROCESS Analysis Data Model: SPECNOATTEMPT;
	data utility.ADaM_SPECNOATTEMPT(compress=yes);
		set utility.ADaM(drop=CTOTHER CTTYPE PSTRATOTH PNONCONOTH PSPECLANG PRSPNDOTH P_COUNT NCTPEROT NCTTELOT SPECLANG STRATOTH RSPNTOTH  FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM CTTYPE CASECONTACT VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(SPECNOATTEMPT) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_SPECNOATTEMPT nodupkeys; by CTRLNUM finaloutcome SPECNOATTEMPT; run;

	%* PROCESS Analysis Data Model: NCTPEROT;
	data utility.ADaM_NCTPEROT(compress=yes);
		set utility.ADaM(drop=CTOTHER CTTYPE PSTRATOTH PNONCONOTH PSPECLANG PRSPNDOTH P_COUNT SPECNOATTEMPT NCTTELOT SPECLANG STRATOTH RSPNTOTH  FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(NCTPEROT) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_NCTPEROT nodupkeys; by CTRLNUM finaloutcome NCTPEROT; run;

	%* PROCESS Analysis Data Model: NCTTELOT;
	data utility.ADaM_NCTTELOT(compress=yes);
		set utility.ADaM(drop=CTOTHER CTTYPE PSTRATOTH PNONCONOTH PSPECLANG PRSPNDOTH P_COUNT SPECNOATTEMPT NCTPEROT SPECLANG STRATOTH RSPNTOTH FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(NCTTELOT) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_NCTTELOT nodupkeys; by CTRLNUM finaloutcome NCTTELOT; run;

	%* PROCESS Analysis Data Model: SPECLANG;
	data utility.ADaM_SPECLANG(compress=yes);
		set utility.ADaM(drop=CTOTHER CTTYPE NCTTELOT PNONCONOTH PSTRATOTH PSPECLANG PSTRATOTH NCTTELOT SPECLANG PRSPNDOTH P_COUNT SPECNOATTEMPT
								NCTPEROT NCTPEROT STRATOTH RSPNTOTH FRDATE -- FRSEC LANGLIST -- NO_NOI_SUM  MARK -- TEXTEMAILIN_OTH
								VER CTTYPE CASECONTACT CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER);
		where not missing(SPECLANG) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_SPECLANG nodupkeys; by CTRLNUM finaloutcome SPECLANG; run;

	%* PROCESS Analysis Data Model: STRATOTH;
	data utility.ADaM_STRATOTH(compress=yes);
		set utility.ADaM(drop=CTOTHER CTTYPE PNONCONOTH PSPECLANG PRSPNDOTH P_COUNT SPECNOATTEMPT NCTPEROT NCTPEROT RSPNTOTH FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(STRATOTH) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_STRATOTH nodupkeys; by CTRLNUM finaloutcome STRATOTH; run;

	%* PROCESS Analysis Data Model: RSPNTOTH;
	data utility.ADaM_RSPNTOTH(compress=yes);
		set utility.ADaM(drop=CTOTHER CTTYPE PNONCONOTH PSPECLANG PSTRATOTH PRSPNDOTH P_COUNT SPECNOATTEMPT NCTPEROT NCTPEROT STRATOTH FRDATE -- FRSEC
							  LANGLIST -- NO_NOI_SUM VER CLANGUAGE_1 -- COMPLETE_THISINT IMMEDIATE_ATTEMPT -- TIMER MARK -- TEXTEMAILIN_OTH);
		where not missing(RSPNTOTH) and not missing(year);
		if missing(finaloutcome) then finaloutcome='999';
	run;
	proc sort data=utility.ADaM_RSPNTOTH nodupkeys; by CTRLNUM finaloutcome RSPNTOTH; run;

	%* calculate number of missing values per variable;
	%count_missing(param=&ds)

	proc sql noprint;
		%do i = 1 %to %sysfunc(countw(&ds, ' '));
			%let holder=%scan(&ds, &i);
			%let holder = %substr(&holder,6); %* remove 'adam_' from dataset name;

			%* get variables from <...>_missing where there are missing values;
			select strip(name) as _n into :names separated by ','
			from utility.&holder._missing
			where count_missing = 0;
			%put ERROR: &holder;
			%put NOTE: &names;

			create table utility.&holder._no_missing as 
			select &names
			from utility.%scan(&ds, &i);
		%end;
	quit;

	
	%* get intermediate, temporary datasets that are to be deleted;
	filename pipefile pipe 'DIR "\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis"';
	data utility.clean_datasets;
		length all_datasets $ 1024;
		length token $ 64;
		retain all_datasets;
		infile pipefile truncover;
		input _stdout_ $char32767.;
		if _n_ = 1 then do;
			ADAMreg = prxparse('/(adam_[a-z]{4,})\.sas7bdat/i');
			MISSreg = prxparse('/([a-z]{4,}_missing)\.sas7bdat/i');

			retain ADAMreg MISSreg;
			call missing( all_datasets, token);
		end;

		if prxmatch(ADAMreg,_stdout_) then do;
			token=prxposn(ADAMreg,1,_stdout_);
			if findw(all_datasets, strip(token)) = 0 then do;
				all_datasets=catx(' ', all_datasets, token);
				call symputx('all_datasets',all_datasets);
				output;
			end;
		end;
		if prxmatch(MISSreg,_stdout_) then do;
			token=prxposn(MISSreg,1,_stdout_);
			if findw(all_datasets, strip(token)) = 0 then do;
				all_datasets=catx(' ', all_datasets, token);
				call symputx('all_datasets',all_datasets);
				output;
			end;
		end;
		call symputx('all_datasets',all_datasets);
		%put NOTE: &all_datasets;
	run;

	* keep only the analysis datasets and the missingness datasets;
	libname utility "&utility";
	proc datasets library=utility noprint;
		save &all_datasets / memtype=data;
	run; quit;
%mend; 
%TextFactory()

libname utility "&utility";
proc download inlib=utility outlib=work memtype=(data); run;
endrsubmit;
signoff profile;
