%macro lib_assign;
	libname b "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\__cache__";
	%do i = 2019 %to 2022;
		%do j = 1 %to 4;
			libname a "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\utility\sas_files\&i.q&j" access =readonly;
			proc copy in=a out=b; run;
			libname a clear;
		%end;
	%end;
%mend; %lib_assign

%let vars = prspndoth rspntoth nctperot;

%macro do_all;
	%do i =1 %to %sysfunc(countw(&vars));
		%let t=%scan(&vars,&i);
		data b.&t(keep=c);
			length c $ 1024;
			set b.q1_&t._nodup b.q2_&t._nodup b.q3_&t._nodup b.q4_&t._nodup;
			c = lowcase(&t);
		run;
	%end;
%mend; %do_all
			
data b.all_vars;
	set b.prspndoth b.rspntoth b.nctperot;
	if countw(c) > 3;
run;

ods html path="\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\corpora" body="allvars.html";
proc print data=b.all_vars noobs; run;
ods html close;
ods html;


	filename o "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\train\allvars.jsonl";
	data _null_;
		length tmp $ 1024;
		set b.all_vars end=last;
		file o lrecl=1024;

		if _n_ = 1 then do;
			regex=prxparse("s/([\'\\\/])//");
			retain regex;
			call missing(tmp);
		end;

		if anyalnum(c)=0 then delete;
		tmp=prxchange(regex,-1,c);

		tmp=compbl(strip(tmp));
		tmp='{"text": "' || strip(tmp) || '"}';
		put tmp;
	run;


		proc surveyselect data=b.all_vars out=b.allvars_shuffled rate=1 outorder=random; run;
