/*
	program to extract QuestionText from Fields table.
	QuestionText to be processed NLP for dependency tree.

	author: alan hamm(pqn7@cdc.gov)
	date: October 2021
*/

%let spiderVersion=Spider-2021Nov;

libname s "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\&spiderVersion\data" access=readonly;
libname o "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\question_text_qc";

options mprint mlogic;
%macro change;
data question_text%substr(&spiderVersion,8)(drop=icon);
	set s.fields(keep=name question_text icon rename=(question_text=QuestionText));

	if _n_ = 1 then do;
		reg1=prxparse('s/\^SCNAME|\^SANAME/the person/i'); 		retain reg1;
		reg2=prxparse('s/\^AreyouIsALIASIsanyone/are you/i'); 	retain reg2;
		reg3=prxparse('s/\^youALIAS/you/i');					retain reg3;
		reg4=prxparse('s/\^areis/are/i');						retain reg4;
		reg5=prxparse('s/\^ALIASNAME/the person/i');			retain reg5;
		reg6=prxparse('s/\d+\s+of\s+\d+/ /i');					retain reg6;
		reg7=prxparse('s/\^yourALIAS/your/i');					retain reg7;
		reg8=prxparse('s/\^youthisperson/your/i');				retain reg8;
		reg9=prxparse('s/\?\s*\[F1\]/ /i');						retain reg9;
		reg10=prxparse('s/\^NOTREPS/not including/i');			retain reg10;
		reg11=prxparse('s/\^hisher_c/her/i');					retain reg11;
		reg12=prxparse('s/\^arrange_callback_fill\d+//i');		retain reg12;
		reg13=prxparse('s/\^dodoes/do/i');						retain reg13;
		reg14=prxparse('s/\^youthey/you/i');					retain reg14;
		reg15=prxparse('s/\^isheshearethey/are they/i');		retain reg15;
		reg16=prxparse('s/\^youthey/you/i');					retain reg16;
		reg17=prxparse('s/\^stayinghere//i');					retain reg17;
		reg18=prxparse('s/\^oncampsa//i');						retain reg18;
	end;

	if not missing(QuestionText) and
		lowcase(icon) = 'question' and
		index(substr(QuestionText,1,2),'**') = 0;

	%do i=1 %to 18;
		call prxchange(reg&i,-1,QuestionText);
		drop reg&i;
	%end;

run;
%mend; %change

proc sort data=question_text%substr(&spiderVersion,8); 
		  by name; 
run;

data o.question_text%substr(&spiderVersion,8); 
	set question_text%substr(&spiderVersion,8)(obs=10);
run;

*libname _all_ clear;
