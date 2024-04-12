				* 			author: alan hamm(pqn7)			*
			*					date: march 2022				*			  						
		*															*
	*															   		*
*																			*
* 														 *
* program to etl multiple f7notes datasets into text file					 *
*****************************************************************************;
	options dlcreatedir;
	libname o_mycsp "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\utility\sas_files\2019q4";
	libname o_mycsp clear;
	libname o_mycsp "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\utility\sas_files";


	libname w "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\utility\sas_files\2021q1" access=readonly;
	libname x "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\utility\sas_files\2021q2" access=readonly;
	libname y "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\utility\sas_files\2021q3" access=readonly;
	libname z "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\nlp\utility\sas_files\2021q4" access=readonly;

	/*proc sort data=y.q3_visits_chi_confidential out=o_mycsp.q3_rspntoth_nodup(keep=rspntoth) nodupkey; by rspntoth; run;*/

	*filename o "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\corpora\2022Q1\rspntoth.jsonl";
	*filename o "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\corpora\rspntoth\rspntoth.txt";
/*
	libname x xml "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\corpora\rspntoth\rspntoth_2022q1.xml";
	data x.rspntoth; set w.rspntoth_nodup(keep=rspntoth); where not missing(rspntoth); run;
	libname x clear;

	data o_mycsp.rspndoth_2021;
		set w.q1_rspntoth_nodup x.q2_rspntoth_nodup y.q3_rspntoth_nodup z.q4_rspntoth_nodup;
	run;

*/
	ods html path = "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\corpora"
		body = "NCTPEROT_2019q4.html";

	data o_mycsp.q4_NCTPEROT_nodup(keep=NCTPEROT);
		set z.q4_NCTPEROT_nodup(keep=NCTPEROT);
		where anyalpha(NCTPEROT) ^= 0;
		if lowcase(NCTPEROT) ^= '& text';
	run;
	proc print data=o_mycsp.q4_NCTPEROT_nodup(keep=NCTPEROT) noobs; run;

	ods html close;
	ods html;


	data _null_;
		length tmp $ 1024;
		set o_mycsp.q3_rspntoth_nodup end=last;
		file o lrecl=256;

		if _n_ = 1 then do;
			regex=prxparse("s/(([\'\\\/])|(\n)|([[:^ascii:]]))//");
			retain regex;
			call missing(tmp);
		end;

		if anyalnum(rspntoth)=0 then delete;
		rspntoth=prxchange(regex,-1,rspntoth);

		tmp=compbl(strip(rspntoth));
		put tmp @@;
		if last then put ' ';
	run;

	filename o "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\corpora\rspntoth2021.jsonl";
	data _null_;
		length tmp $ 1024;
		set o_mycsp.rspndoth_2021 end=last;
		file o lrecl=256;

		if _n_ = 1 then do;
			regex=prxparse("s/(([\'\\\/])|(\n)|([[:^ascii:]]))//");
			retain regex;
			call missing(tmp);
		end;

		if anyalnum(rspntoth)=0 then delete;
		rspntoth=prxchange(regex,-1,rspntoth);

		tmp=compbl(strip(rspntoth));
		put tmp @@;
		if last then put ' ';
	run;


	data _null_;
		length tmp $ 1024;
		set o_mycsp.rspndoth_2021 end=last;
		file o lrecl=1024;

		if _n_ = 1 then do;
			regex=prxparse("s/[\'\\\/]//");
			retain regex;
			call missing(tmp);
		end;

		if anyalnum(rspntoth)=0 then delete;
		rspntoth=prxchange(regex,-1,rspntoth);

		tmp=compbl(strip(rspntoth));
		tmp='{"text": "' || strip(rspntoth) || '"}';
		put tmp;
	run;
	filename o clear;





options dlcreatedir;
libname o_mycsp "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\utility\sas_files\20219q4";
libname o_mycsp clear;
libname o_mycsp "\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\NLP\utility\sas_files\20219q4";

libname a "\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_MASTER\PII_Archive\2019\Q1\Level2" access=readonly;
libname b "\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_MASTER\PII_Archive\2019\Q2\Level2" access=readonly;
libname c "\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_MASTER\PII_Archive\2019\Q3\Level2" access=readonly;
libname d "\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_MASTER\PII_Archive\2019\Q4\Level2" access=readonly;

proc sort data=d.q4_visits_chi_confidential(where=(NCTPEROT^=' ')) out=o_mycsp.q4_NCTPEROT_nodup(keep=NCTPEROT) nodupkeys; by NCTPEROT; run;
