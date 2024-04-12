
********* Configure these as needed ***********;

%let filename=adult;  * adult | child, CSPMID.Variable table;
%let SurveyYear=2020; * CSPMID.Variable table;	
%let lvl3quarter=4;	  * level 3 datafile quarter;
%let er=;
%let quarter=4; 	  * spider quarter MUST MATCH SPIDERVERSION;
%let spiderversion=spider-2020nov;


************ do not edit below ************;
options nomprint nomlogic nosymbolgen;
options mprint mlogic symbolgen notes;

* create libref to CSPMID library;
libname mid odbc dsn='CSPMID' schema='NHISMetaData' access=readonly;

* set macro library;
options mautosource sasautos=("\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\GitRepository\Code\Library\", sasautos);

* set production parameters;
%setProductionParameters(year=&SurveyYear, quarter=&quarter,spiderversion=&spiderversion)

* macro to ETL data from CSPMID.Variable;
%macro etlmid(filename=, surveyyear=, spiderversion=, inhpub=);
	%global verbatimnames;
	%* to subset CSPMID.SectionInfo table;
	%if &quarter = 1 %then %let _quarter= spider-&SurveyYear((jan)|(feb)|(march));
	%else %if &quarter = 2 or &quarter = 1 2 %then %let _quarter= spider-&SurveyYear((apr)|(may)|(jun));
	%else %if &quarter = 3 or &quarter = 1 2 3 %then %let _quarter= spider-&SurveyYear((july)|(aug)|(sep));
	%else %if &quarter = 4 or &quarter=%str() %then %let _quarter= spider-&SurveyYear((oct)|(nov)|(dec));

	proc sql;
		create table _variable(compress=yes) as 
		select SpiderVersion,	VarType label='InstrumentVarType', module, section,

				case L3VariableName
				when ' ' then put(Name, $32.)
				else L3VariableName
				end as Name length=32 label='L3VariableName',

				Description, LeaveAsChar label='Character or Numeric', QuestionText, QuestionID, 
				catx(', ',strip(GenericEdit1), strip(GenericEdit2), strip(GenericEdit3)) as GenericEdits,
				catx(', ',strip(OutputFileName), strip(OutputFileName2), strip(OutputFileName3), strip(OutputFilename4)) as OutputFileNames, FileName,
				VariableGroup, ImpDec, OutputAnswerList length=1024, OutputCodes label='ResponseCode Value(s) Set',
				OutputLength informat=best32. format=best32.
		from mid.Variable
		where prxmatch("/&_quarter/i",SpiderVersion) and 
			  prxmatch("/\b&inhpub\b/i", VariableGroup) and
			  (prxmatch("/&filename/i", filename) )
		order by varType, module, section, name;

		data _variableNumeric 				%* include where output code contains only numeric response codes;
			 _VariableVerbatim 				%* include where output code contains 'verbatim' response code;
			 _VariableNonVerbatim 			%* include where output code contains non 'verbatim' alpha response codes;
			 _VariableMissingOutputCode;	%* include where output code is missing;
			set _variable;

			%* transform range response code containing '-' to ':' for use with IN operator;
			if indexc(OutputCodes,'-') then do;
				OutputCodes=translate(OutputCodes,':','-');
			end;
		
			if missing(OutputCodes) then output _VariableMissingOutputCode;
			else if indexw(lowcase(OutputCodes),'verbatim', ' ,') then output _VariableVerbatim;
			else if anyalpha(OutputCodes) then output _VariableNonVerbatim;
			else if anyalpha(OutputCodes)=0 and missing(LeaveAsChar) then output _variableNumeric;
		run;

		%* add F7note Name to _VariableVerbatim table;
		data _VariableVerbatim;
			set _VariableVerbatim end=last;
			output;
			if last then do;
				call missing(varType, module, section, description, leaveAsChar, questionText, questionId, genericEdits);
				call missing(outputfilenames, filename, variablegroup, impdec, outputAnswerList, OutputCodes, Outputlength);
				name='F7Note';
				OutputCodes='verbatim';
				output;
			end;
		run;

	%* assign variables with 'verbatim' response code to macro variable;
	proc sql noprint;
		select name as _name into :verbatimnames separated by ' '
		from _variableverbatim;
		/*%put &verbatimnames;*/
	quit;
	%put _user_;
%mend etlmid;
%etlmid(filename=&filename,surveyyear=&SurveyYear, spiderversion=&spiderversion, inhpub=l3)


* remove quote length max of 256 characters;
options noquotelenmax notes;
%macro combinelvl3;
	
	%* create libref to level3 datafile library;
	%* NEED TO INCLUDE CHECK FOR MISSING QUARTER DATAFILES;
	%* VALIDATE THAT USING QUARTERLY FILE IS MORE EFFICIENT THAN MONTHLY LEVEL3;
	%do i = 1 %to 4;
		libname level3 "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&SurveyYear\Q&i\level3" access=readonly;
		proc sql noprint;

			* get level 3 data file; 
			create table q&i._level3_temp1(compress=yes) as 
			select *
			from level3.q&i._sample_&filename&er;

			create table q&i._level3_temp2(compress=yes) like q&i._level3_temp1;

			* get level 3 variable metadata;
			create table q&i._lvl3var_temp1(compress=yes) as 
			select *
			from dictionary.columns
			where lowcase(libname)="work" and lowcase(memname)="q&i._level3_temp1";

			create table q&i._lvl3var_temp2(compress=yes) like q&i._lvl3var_temp1; 
		quit;

		proc sort data=q&i._level3_temp1; by acaseid; run;
		proc sort data=q&i._level3_temp2; by acaseid; run;
		data q&i.level3(compress=yes);
			merge q&i._level3_temp2 q&i._level3_temp1;
			by acaseid;
		run;

		proc sort data=q&i._lvl3var_temp1; by name; run;
		proc sort data=q&i._lvl3var_temp2; by name; run;
		data q&i.lvl3var(compress=yes);
			merge q&i._lvl3var_temp2 q&i._lvl3var_temp1;
			by name;
		run;
		libname level3 clear;
	%end;

	data level3;
		set q1level3 q2level3 q3level3 q4level3;
		by acaseid;
	run;
	proc sort data=level3; by acaseid; run;

	data lvl3var;
		set q1lvl3var q2lvl3var q3lvl3var q4lvl3var;
		by name;
	run;
	proc sort data=lvl3var nodupkeys; by name; run;

	proc datasets noprint;
		delete q1level3; delete q2level3; delete q3level3; delete q4level3; 
		delete q1lvl3var; delete q2lvl3var; delete q3lvl3var; delete q4lvl3var;
		delete q1_level3_temp1; delete q2_level3_temp1; delete q3_level3_temp1; delete q4_level3_temp1;
		delete q1_level3_temp2; delete q2_level3_temp2; delete q2_level3_temp2; delete q3_level3_temp2; delete q4_level3_temp2;
		delete q1_lvl3var_temp1; delete q2_lvl3var_temp1; delete q3_lvl3var_temp1; delete q4_lvl3var_temp1;
		delete q1_lvl3var_temp2; delete q2_lvl3var_temp2; delete q2_lvl3var; delete q3_lvl3var_temp2; delete q4_lvl3var_temp2;
	run; quit;
%mend combinelvl3; %combinelvl3


/**********************************************************
* extract level 3 datafile data where output code response 
* code contains 'verbatim'
**********************************************************/

* macro to extract F7notes variable values from monthly level3 data;
%macro etlF7;
	%*create empty table for subsequent match-merge with created monthly F7 subset dataset;
	proc sql; create table _f7_ (Name char(32), ACASEID char(10), F7Note char(1000)); quit;

	%* iterate over 12 months of level 3 monthly datafiles to extract F7note dataset;
	%* needs modification to reduce code;
	%do i = 1 %to 12;
		%* month 1 to 9;
		%if %eval(&i < 10) %then %do;
			libname _t "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&SurveyYear\M0&i\level3" access=readonly;
			data m0&i.F7(rename=(variable_name=name));
				set _t.m0&i._f7note(keep=acaseid f7note variable_name);
			run;

			proc sort data=_f7_; by name; run;
			proc sort data=m0&i.F7; by name; run;
			data m&i._f7notes;
				merge _f7_ m0&i.F7;
				by name;
			run;
			libname _t clear;
		%end;
		%* month 10 to 12;
		%else %if %eval(&i >= 10) %then %do;
			libname _t "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&SurveyYear\M&i\level3" access=readonly;
			data m&i.F7(rename=(variable_name=name));
				set _t.m&i._f7note(keep=acaseid f7note variable_name);
			run;
			proc sort data=_f7_; by name; run;
			proc sort data=m&i.F7; by name; run;
			data m&i._f7notes;
				merge _f7_ m&i.F7;
				by name;
			run;
			libname _t clear;
		%end;
		proc sort data=m&i._f7notes; by name; run;
	%end;

	data _f7notes(compress=yes);
		set 
			%do i=1 %to 12; m&i._f7notes %end;
		;
		by name;
	run;

	proc datasets noprint;
		%do i = 1 %to 12;
			delete m&i._f7notes;
		%end;
	run; quit;
%mend etlf7; %etlf7

* match-merge level3 datafile with _f7notes dataset;
proc sort data=level3; by acaseid; run;
proc sort data=_f7notes; by acaseid; run;

data _level3_f7(compress=yes);
	merge level3(in=a) _f7notes(drop=name in=b);
	by acaseid;
run;

* sort and keep on those variables where the output codes contains response code of 'verbatim';
proc sort data=_level3_f7(keep=&verbatimnames); by acaseid; run;
%put _user_;


/* GENERATE CORPUS FILES */

* corpus library;
libname o "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\Level3Corpus\SASds" outencoding=utf8;

* generate category information;
data o.hispother_a(keep=hispother_a where=(hispother_a ^= ''))
	 o.HISPVRBAT_A(keep=HISPVRBAT_A where=(HISPVRBAT_A ^= ''))
	 o.PIOTHER_A(keep=PIOTHER_A where=(PIOTHER_A ^= ''))
	 o.pivrbat_a(keep=pivrbat_a where=(pivrbat_a ^= ''))
	 o.asianother_a(keep=asianother_a where=(asianother_a ^= ''))
	 o.asianvrbat_a(keep=asianvrbat_a where=(asianvrbat_a ^= ''))
	 o.raceother_a(keep=raceother_a where=(raceother_a ^= ''))
     o.racevrbat_a(keep=racevrbat_a where=(racevrbat_a ^= ''))
	 o.EMDWHOWRK_A(keep=EMDWHOWRK_A where=(EMDWHOWRK_A ^= ''))
	 o.EMDKINDWRK_A(keep=EMDKINDWRK_A where=(EMDKINDWRK_A ^= ''))
	 o.EMDKINDIND_A(keep=EMDKINDIND_A where=(EMDKINDIND_A ^= ''))
     o.EMDIMPACT_A(keep=EMDIMPACT_A where=(EMDIMPACT_A ^= ''))
	 o.f7note_a(keep=f7note where=(f7note ^= ''));
	set _level3_f7(keep=hispother_a -- f7note);
run;


* create categories sas dataset;
proc sql;
	create table o.manifest as 
	select *
	from dictionary.tables
	where lowcase(libname) = 'o';
quit;

proc sql;
	create table o.feed as
	select *
	from dictionary.columns
	where lowcase(libname) = 'o';
quit;


proc json out="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\Level3Corpus\manifest.json" pretty nosastags nofmtcharacter nofmtdatetime nofmtnumeric nosastags scan;
	export o.manifest ;
run;



proc json out="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\Level3Corpus\feed.json" pretty nosastags nofmtcharacter nofmtdatetime nofmtnumeric nosastags scan;
	export o.feed ;
run;


/*
filename in "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\Level3Corpus\feed.json";
filename map "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\Level3Corpus\feed.map";
libname in json map=map automap=replace;
proc datasets library=in ; run; quit;
filename in clear; 
filename map clear;
libname in clear;

ods html path="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\Level3Corpus\hispother_a" 
		 body="feed.html" style=SASWeb;

proc datasets lib=o; run; quit;

ods html close;

ods listing;
filename in "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\Level3Corpus\hispother_a\feed.json";
filename map "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\Level3Corpus\hispother_a\feed.map";
libname in json map=map automap=create;
proc contents data=in._all_; run;
*/
