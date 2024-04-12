/* getF7notes.sas
   
   Description: ELT F7 notes from Preview level 3
   Author: Alan Hamm(pqn7@cdc.gov)
   Date: June 2021
*/
options mprint mlogic;
%let surveyyear=2021;
%let month=07;

*****************************************************************************************;
%let preview=\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&SurveyYear\M&month\Preview;
%let NLPfolder=\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\training_exercises;

* get most recent preview level 3 folder;
filename preview pipe "DIR &preview /A:D /T:C /O:-D";
data _null_;
	length _stdout_ $ 32000 previewWeek $ 32;
	infile preview truncover;
	input _stdout_ $char32000.;
/*
	if _n_ = 1 then do;
		parse=prxparse('/\d\d\/\d\d\/\d{4}\s*\d\d:\d\d\s*AM|PM\s*<DIR>([A-Z0-9][\-])/i');
		retain parse;
	end;

	if prxmatch(parse,_stdout_);
*/
	if indexw(_stdout_,'<DIR>') and not index(_stdout_,'level') and not indexc(_stdout_,'.');
	previewWeek=scan(_stdout_,5,' ');
	call symputx('previewWeek',previewWeek);
	output;
	stop;	
run;
%put NOTE: Preview Week extracted from Preview\Level3 folder: &previewWeek;

options dlcreatedir;
libname _t "&NLPfolder\M&month._&previewWeek";
libname _t clear;
	
* extract F7 note datafile;
*libname _t "&preview\&previewWeek\level3" access=readonly;
libname _t "&preview\July1-30\level3" access=readonly;

filename _out "&NLPfolder\F7NoteOutput.txt";
*filename _out "&NLPfolder\M&month._&previewWeek\F7NoteOutput.txt";
data m&month._f7notes(keep=text /*language delivery id*/);
	*retain ID 0;
	length text $ 5000;
	retain language 'en';
	retain delivery "&previewWeek";

	file _out;
	set _t.m&month._f7note(keep=f7note rename=(f7note=text));

	if _n_=1 then do;
		regASCII=prxparse('s/[[:^ascii:]]//');
		regLINE=prxparse('s/(([\n]*)|([\r]*))//');
		retain regASCII regLINE;
	end;

	* remove non-ASCII characters;
	call prxchange(regASCII,-1,text);

	* remove new-line and return characters;
	call prxchange(regLINE,-1,text);


	* write to txt file;
	put text;
	
	* write to dataset for proc export csv;
	*text=text||'\n';
	output;
run;

proc export data=m&month._f7notes
		    outfile="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\training_exercises\F7NoteOutput.txt"
			dbms=tab
			replace;
run;
/*
proc export data=m&month._f7notes
		    outfile="&NLPfolder\M&month._&previewWeek\F7NoteOutput.csv"
			dbms=csv
			replace;
run;
*/

* write jsonl;
*filename _out "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\production_data\text_sentences.jsonl";
filename _out "&NLPfolder\F7NoteOutput.jsonl";
data m&month._f7notes(keep=text);
	*retain ID 0;
	length text $ 5000;
	retain language 'en';
	retain delivery "&previewWeek";

	length tmp $ 5000;

	file _out;
	set _t.m&month._f7note(keep=f7note rename=(f7note=text));

	if _n_=1 then do;
		reg=prxparse('s/[[:^ascii:]]//');
		retain reg;
	end;

	if anyalpha(text)=0 then delete;

	* remove non-ASCII characters;
	call prxchange(reg,-1,text);
	
	text=translate(text,trimn(''),'"');

	* write to txt file;
	tmp='{"text":'|| '"' || strip(dequote(text)) || '."}';
	put tmp;
run;


* write JSON file;
proc json out="&NLPfolder\m&month._&previewWeek\F7NoteOutput.json" nosastags pretty;
	export m&month._f7notes(keep=text);
run;
	
libname _t clear;
