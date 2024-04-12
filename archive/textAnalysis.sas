* author: alan hamm
  date: october 2022

  description: data profiling and text analysis of textFactory.sas which is to be executed first
;
%let utility = \\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis;
%let sentDIR = 'DIR "\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis"'%str(;);

libname utility "&utility";

%let ds=adam_ctother adam_nctperot adam_ncttelot adam_pnonconoth adam_prspndoth
		adam_pspeclang adam_pstratoth adam_rspntoth adam_specnoattempt adam_stratoth;


%* execute batch file to use python to execute sentiment analysis, and output CSV files;
libname utility "&utility";
filename sent pipe '\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\GitRepository\Code\NLP\sentiment.bat';
data _null_;
	infile sent truncover;
	input _stdout_ $char32767.;
run;


%* write output from sentiment.py to sas7bdat filetype;
%macro import_csv(file=%bquote(&ds), dir=%bquote(&utility));
	%do index=1 %to %sysfunc(countw(&ds, ' '));
		%let tmp_name=%substr( %scan(&ds, &index), 6);
		libname utility "&dir";
		proc import datafile="&dir\&tmp_name._sentiment.csv"
					out=utility.&tmp_name._sentiment
					dbms=csv replace;
		run;
		data utility.&tmp_name._sentiment; format _all_; set utility.&tmp_name._sentiment;  run;
	%end;
%mend; %import_csv()


%macro do_freq(files=&ds);
	proc format;
		value $miss_sleuth
			' ' = 'Missing String'
			other = 'Text String';

		value $sentiment
			'positive' = 'Positive Sentiment'
			'negative' = 'Negative Sentiment'
			'neutral' = 'Neutral Sentiment'
			other = 'Error in Sentiment Proc Format';

		value sent_score
			low - 0.200 = '-1 to 0.199'
			0.200 - 0.300 = '0.200 to 0.299'
			0.300 - 0.400 = '0.300 to 0.399'
			0.400 - 0.500 = '0.400 to 0.499'
			0.500 - 0.600 = '0.500 to 0.599'
			0.600 - 0.700 = '0.600 to 0.699'
			0.700 - 0.800 = '0.700 to 0.899'
			0.900 - 1 = '0.900 to 1';
	run;
	%do index=1 %to %sysfunc(countw(&files, ' '));
		%let file_name=%substr( %scan(&files, &index), 6 );
		title "Text Profile for %upcase(&file_name)";
		/*proc sort data=utility.&file_name._sentiment; by &file_name; run;*/
		proc freq data=utility.&file_name._sentiment;
			tables &file_name / missing;
			format &file_name $miss_sleuth. sentiment $miss_sleuth.;

			tables sentiment / missing;
			format sentiment $sentiment.;

			tables &file_name * sentiment / missing;
			format &file_name $miss_sleuth. sentiment $sentiment.;
			
			tables pos * sentiment / missing;
			format pos sent_score. sentiment $sentiment.;

			tables neg * sentiment / missing;
			format neg sent_score. sentiment $sentiment.;

			tables outcome*(year sentiment);
		run;
	 %end;
%mend; %do_freq()


%macro get_files();
	
	filename sent pipe &sentDir;
	data utility.sentiment_files(drop=regex);
		infile sent truncover;
		input _stdout_ $char32767.;

		format datafile $char64. all_datafiles $char1000.;
		retain all_datafiles;

		if _n_ = 1 then do;
			regex = prxparse('/([a-z0-9]+)\_sentiment\.sas7bdat/i');
			retain regex;
		end;

		if prxmatch(regex,_stdout_) then do;;
			datafile=prxposn(regex, 1, _stdout_);
			if findw(all_datafiles, datafile) = 0 then all_datafiles=catx(' ', strip(all_datafiles), strip(datafile));
			call symputx('all_datafiles', all_datafiles);
			output;
		end;
	run;
	%put NOTE: The datafiles -- &all_datafiles;
%mend; %get_files()


* Measures of Central Tendency: The Mean Vector
  The sameple covariate is a measure of the associate between a paire of variables.
;

%macro get_vars(lib=, d=);
	%global names;
	proc sql noprint;
		select name into :names separated by ' '
		from dictionary.columns
		where lowcase(libname)="%lowcase(&lib)" and lowcase(memname)="%lowcase(&d._sentiment)"
			and lowcase(type)='num'
		order by name;
		%put NOTE: &names;
	quit;
%mend;


%macro central_tendency(dsname=&ds);
	%do index=1 %to %sysfunc(countw(&dsname, ' '));
		%let tmp_name=%substr(%scan(&dsname, &index), 6);
		%get_vars(lib=utility, d=&tmp_name)
		title "Central Tendency: &tmp_name";
		proc means data=utility.&tmp_name._sentiment n mean max min range std fw=8;
			var &names;
		run;
		title;
	%end;
%mend; %central_tendency()


%macro cov_matrix(dsname=&ds);
	%do index=1 %to %sysfunc(countw(&dsname, ' '));
		%let tmp_name=%substr(%scan(&dsname, &index), 6);
		%get_vars(lib=utility, d=&tmp_name)
		%put WARNING: &tmp_name;
		title "Covariance Matrix: &tmp_name";
		ods graphics on;
		ods select Cov PearsonCorr;
		proc corr data=utility.&tmp_name._sentiment
			/*nocorr*/
			out=utility.&tmp_name._outCov(type=cov)
			nomiss cov;
			var &names;
		run;

		proc sgscatter data=utility.&tmp_name._sentiment;
			matrix 	pos neg neu /group=outcome diagonal=(histogram kernel);
		run;
		title;
	%end;
%mend; %cov_matrix()












/*
%macro association(dsname=&ds);
	%do index=1 %to %sysfunc(countw(&dsname, ' '));
		%let tmp_name=%substr(%scan(&dsname, &index), 6);
		%get_vars(lib=utility, d=&tmp_name)
		%put WARNING: &tmp_name;
		title "Principal Component: &tmp_name";
		proc princomp data=utility.&tmp_name._sentiment;
			var &names;
		run;
		title;
	%end;
%mend; %association()
*/



/* 
only for univariate?? I think so.
%macro test;
options nospool;
ods graphics on;

proc glm data = utility.ctother_sentiment plots=all;
	class finaloutcome;
	model outcome = &names;
run; quit;
ods graphics off;
%mend; %test
*/
