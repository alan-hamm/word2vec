/*

	Author: Alan Hamm(pqn7@cdc.gov)
	Date:	June 2021					
*/

********* Configure these as needed ***********;

%let filename=adult;  * adult | child, CSPMID.Variable table;
%let year=2021;		  * CSPMID.Variable table;	
%let quarter=2;
%let month=m06;

*preview data, each subsequent week is combined with the present week;
%let prevPreview=July1-11; 
%let presPreview=Jun1-19;

libname prev "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&year\&month\Preview\&prevPreview\level3" access=readonly;
libname pres "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&year\&month\Preview\&presPreview\level3" access=readonly;

/* not needed. all unique acaseid are in previous and present datafiles */
proc sort data=prev.&month._f7note out=a_F7note(keep=acaseid section_name variable_name f7note rename=(section_name=section variable_name=Name)); by acaseid; run;
proc sort data=prev.&month._f7note out=b_F7note(keep=acaseid section_name variable_name f7note rename=(section_name=section variable_name=Name)); by acaseid; run;
data f7note;
	merge a_F7note(in=a) b_F7note(in=b);
	by acaseid;
run;

/*
proc json out="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2021\QualityControl\M05_MAY1-29\F7NoteOutput.json" nofmtcharacter nofmtdatetime nofmtnumeric nosastags;
   export work.f7note(keep=acaseid f7note)  ;
run;
*/

filename _tmp_ "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2021\QualityControl\M06_JUN1-12\F7NoteOutput.txt";
data _null_;
	set work.f7note(keep=acaseid name f7note);

	if _n_=1 then do;
		regASCII=prxparse('s/([^\0-\177])//');
		retain regASCII;
	end;

	/* remove non-ascii characters */
	if prxmatch(regASCII,f7note) then do;
		*putlog "WARNING: " f7note;
		length temp $ 256;
		temp=prxposn(regASCII,0,f7note);
		putlog "WARNING: ASCII CATCH " acaseid= temp=;
		/* remove non-ascii characters 0 - 127 */
		f7note=prxchange(regASCII,-1,f7note);
	end;
	file _tmp_;
	put acaseid +(-1) "::" name +(-1) ":: " f7note;
run;
