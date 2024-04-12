%macro create_dictionary;

	libname a "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora";	
	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\manual_correction.xlsx"
		dbms = excel
		out = a.manual_correction
		replace;
	run;

	/*
	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\action_verbs.xlsx"
		dbms = excel
		out = a.action_verbs
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\adverbs.xlsx"
		dbms = excel
		out = a.adverbs
		replace;
	run;


	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\BuckleySaltonSWL.xlsx"
		dbms = excel
		out = a.BuckleySaltonSWL
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\BuckleySaltonSWL.xlsx"
		dbms = excel
		out = a.BuckleySaltonSWL
		replace;
	run;


	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\Dolch_words.xlsx"
		dbms = excel
		out = a.Dolch_words
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\Fry_1000.xlsx"
		dbms = excel
		out = a.Fry_1000
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\Dolch_words.xlsx"
		dbms = excel
		out = a.dolch_words
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\Grady_Augmented.xlsx"
		dbms = excel
		out = a.Grady_Augmented
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\negative_words.xlsx"
		dbms = excel
		out = a.negative_words
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\OnixTxtRetToolkilSWL1.xlsx"
		dbms = excel
		out = a.OnixTxtRetToolkilSWL1
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\power_words.xlsx"
		dbms = excel
		out = a.power_words
		replace;
	run;


	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\prepositions.xlsx"
		dbms = excel
		out = a.prepositions
		replace;
	run;


	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\strong_words.xlsx"
		dbms = excel
		out = a.strong_words
		replace;
	run;

	proc import
		datafile = "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora\submission_words.xlsx"
		dbms = excel
		out = a.submission_words
		replace;
	run;
	*/


	data a.dictionary(drop=f1);
		length word $ 64;
		label word = ' ';
		set a.action_verbs(rename=(x=word))
			a.adverbs(rename=(v1=word))
			a.amplification_words(rename=(x=word))
			a.buckleySaltonSWL(rename=(x=word))
			a.deamplification_words(rename=(x=word))
			a.dolch_words(rename=(x=word))
			a.fry_1000(rename=(x=word))
			a.Grady_Augmented(rename=(x=word))
			a.negative_words(rename=(x=word))
			a.OnixTxtRetToolkilSWL1(rename=(x=word))
			a.power_words(rename=(x=word))
			a.prepositions(rename=(x=word))
			a.strong_words(rename=(x=word))
			a.submission_words(rename=(x=word))
			a.manual_correction(rename=(x=word) keep = x);
	run;

	proc sort data=a.dictionary out=a.dictionary_clean nodupkeys; by word; run;
	libname a clear;
%mend create_dictionary;

/*
filename contents pipe "DIR \\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\DictionaryCorpora /A:-D" LRECL=2000;

data contents;
	length _stdout_ $ 2000 xlsxfile $ 64;

	
	if _n_ = 1 then do;
		regex = prxparse('/([a-z_]+\.xlsx)/i');
		retain regex;
	end;

	infile contents truncover;
	input _stdout_ $char2000.;

	if prxmatch(regex,_stdout_);
	xlsxfile = prxposn(regex,1,_stdout_);
run;
*/
