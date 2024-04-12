/* author Alan Hamm(pqn7@cdc.gov)
   date August 2021

   description: amalgamate self-assessed number F7note datasets across year(s), quarter(s), months(s)
 			    parse each F7note row first names
				names defined in
				\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\first_names.csv

				Harvard Dataverse( https://dataverse.harvard.edu/ )
					Version 1.3
					Data for: Demographic aspects of first names
						https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TYJKEZ
						https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/TYJKEZ/MPMHFE&version=1.3

					Description 
					The list includes 4,250 first names and information on their respective count and proportions across 
						six mutually exclusive racial and Hispanic origin groups. These six categories are consistent with 
						the categories used in the Census Bureau's surname list. (2017-11-14)
					Subject: Social Sciences
					Keyword: Race and ethnicity, first name.
					Citation: Tzioumis, Konstantinos (2018) Demographic aspects of first names, Scientific Data, 5:180025 [dx.doi.org/10.1038/sdata.2018.25].
*/

%* used in MACRO alphabetsoup;
%global alphabet;
%let alphabet=a b c d e f g h i j k l m n o p q r s t u v w x y z;


options nomlogic nomprint nosymbolgen NOQUOTELENMAX;
*options mlogic mprint symbolgen;

%* update names if macvar update_names = 1;
%macro update_name_list(T=1);
	%if %eval(&t=1) %then %do;
		proc import datafile="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\first_names.csv"
			dbms=csv
			out=harvard_firstnames
			replace;
		run;

		data harvard_firstnames;
			length first_letter $ 1;
			set harvard_firstnames;
			first_letter=first(firstname); /* not needed, also adjust below PROC SORT BY stmt */

			/* temporarily remove some names such as 'In' or 'Do' that require significant */
			/* time-resource bc of visual & cleaning of program output file */
			if lengthn(firstname) >= 5;
			/* temporarily remove name He  as F7 notes use pronoun He with no */
			/* seeming(visual check and cleaning) apparent use of name Said    */
			if lowcase(firstname) ^= 'he';
		run;
		proc sort data=harvard_firstnames; by first_letter; run;
		libname a "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data";
		proc copy in=work out=a;
			select harvard_firstnames;
		run;
		libname a clear;
		libname a "\\cdc.gov\project\CCHIS_NCHS_DHIS\HIS_ALL\Databases\NLP";
		proc copy in=work out=a;
			select harvard_firstnames;
		run;
		libname a clear;
	%end;
	%else %do;
		%put NOTE: No update to First Name dictionary;
		%put;
	%end;
%mend; %update_name_list()
%* end update names;


/* combine all years and use proc survey select to get sample for macro sring_shredder */
%macro combine;

	/* 9.1.2021 FOR VALIDATION ONLY USING JAN, FEB F7notes */

	/* instert algorithm to delete all temporary datasets...think when combining multiple years */
	%do i = 3 %to 4;
		libname m0&i "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2021\M0&i\Preview\level3" access=readonly;
		data mo&i;
			length f7note $ 32767;
			set m0&i..m0&i._f7note(keep=variable_name acaseid f7note rename=(VARIABLE_NAME=var )where=(not missing(f7note)) );
			if not missing(compress(f7note,,'kda'));
			/* remove double quotes */
			f7note=strip( prxchange('s/[\"]//',-1, f7note) );
		run;

		%if %eval(&i=4) %then %do;
			data month_combine(compress=yes);
				set 
				%do i = 3 %to 4;
					mo&i
					%if %eval(&i=4) %then %do;
						%let end_semi_colon=%str(;);
						&end_semi_colon;
					%end;
				%end;
			run;
		%end;
	%end;

	%* exploratory analysis;
	proc sql noprint;
	/*proc sql;*/
		select f7note, count(f7note)
		into :aa, :bb
		from month_combine
		group by  f7note
		having count(*) > 1;

		%*remove leading blanks;
		%let aa=&aa; %let bb=&bb;
	quit;
	%if %eval(&aa ^= %str()) or  %eval(&bb ^= %str()) %then %do;
		title1 "Duplicate values found SORT NODUPKEY: f7note";
		%PUT WARNING: FOUND DUPLICATE BY KEYS %str(%')F7NOTE%str(%');
		%PUT WARNING: ACASEID COUNT: &bb VARIABLE COUNT: &aa F7Note COUNT: &bb;
		%PUT WARNING: DUPLICATES ARE REMOVED(see Results Window or work.month_combine_duplicates);
		proc sort data=month_combine dupout=month_combine_duplicates(label='Duplicate Keys: F7NOTE') nodupkeys; by f7note; run;
		title;
	%end;
	libname _all_ clear;
%mend; %combine

options nonotes;
%* push firstnames into macvars variables: name_<a,b,c,d,...,z>; 
%macro alphabetsoup;
	libname a "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data" access=readonly;

	%* create global name macvars for subsequent macvar gen;
	%do i = 1 %to 26; %global name_%scan(&alphabet,&i); %end;
		
	%do i= 1 %to 26;
		proc sql noprint;
			select cats('(\b',strip(firstname),'\b)') as name_%scan(&alphabet,&i)
			into :name_%scan(&alphabet,&i) separated by '|'
			from a.harvard_firstnames
			where indexc( lowcase(first_letter), scan("&alphabet",&i))
			order by name_%scan(&alphabet,&i);
		quit;
	%end;
	libname a clear;
%mend; %alphabetsoup


options notes nomprint nomlogic nosymbolgen NOQUOTELENMAX ;
*options notes mprint mlogic nosymbolgen NOQUOTELENMAX ;

/* set aaa = 1 to write data to permanent location */
%macro string_shredder(aaa=1);

	* defensive programming;
	proc sort data=month_combine; by f7note; run;

	data f7notes_2(compress=yes keep=f7note acaseid var);

		if _n_=1 then do;

			/* temporary f7note holder to ensure leading/trailing blanks removed */
			/* if they aren't removed, distorts position of found entities.		 */
			length tmp_f7 $ 32767;

			/* variables to hold names*/
			length first_name $ 128 first_name_list $ 10000;

			/* variables to hold labels(e.g. (0, 5, 'PERSON') */
			length entities $ 5000  tmp_entities $ 5000;

			/* variable to hold f7note and entities 						 */
			/* e.g. ("Uber blew through $1 million a week", [(0, 4, 'ORG')]) */
			length _l $ 32767;

			call missing(entities, tmp_entities, tmp_f7, first_name, first_name_list, _l);

			/* hash keys and data */
			length k $ 32767;	/* f7 note */
			length s $ 5000;	/* appeneded, full entity list */
			length f $ 10000;	/* appended, full first name list */
			length _e $ 5000;	/* hold single entity */
			length _n $ 127; 	/* first name */
			length _k $ 32767; 	/* f7 strings */

			/* hash table to map a F7notes to a fully extracted and processed PERSON entities */
			declare hash entity_list(multidata:'y');
			entity_list.defineKey('k');
			entity_list.defineData('k','s','f','_l');
			entity_list.defineDone();
			declare hiter iter('entity_list');
			call missing(k,s,f,e);

			/* clear hash entity_list */
			rc=entity_list.clear();

			declare hash tmp_name_hash(multidata:'y');
			tmp_name_hash.defineKey('_k', '_e');
			tmp_name_hash.defineData('_k','_e','_n','_l');
			tmp_name_hash.defineDone();
			call missing(_k,_e,_n);
			declare hiter tmpiter('tmp_name_hash');

			/* another hash to just hold the f7 strings */
			declare hash varhash(dataset: ' month_combine');
			varhash.defineKey('f7note');
			varhash.defineDone();
		end;

		set month_combine /*(obs=200)*/ end=last ;
		by f7note;

		start=1;
		tmp_f7=strip(f7note);
		stop=lengthn(tmp_f7);

		%do i=1 %to 26;

			%let qwerty=%scan(&alphabet,&i);
			%let tmp="/&&name_&qwerty/i";
			regex=prxparse(&tmp);

			call prxnext(regex,start, stop, tmp_f7, position, length);
		 	do while (position > 0 );

					first_name=substr(strip(tmp_f7),position,length);
					first_name_list=catx(', ', strip(first_name_list), strip(first_name) );

					tmp_entities='(' || strip(put(position-1,best16.)) || ', ' || strip(put(lengthn(first_name),best16.)) || ", 'PERSON')";

					_l="('" || strip(tmp_f7) || "', " || "{'entities': " || '[' || strip(tmp_entities) || ']})';

					tmp_name_hash.add(key: f7note, key: tmp_entities, data: tmp_f7, data: tmp_entities, data: first_name, data: _l);

					if not missing(tmp_entities) then do;
				    	if varhash.check() =0  then do;
							 if entity_list.find(key: tmp_f7) = 0 then do;
								entities=catx(', ', strip(entities), strip(tmp_entities));
								_l="('" || strip(tmp_f7) || "', " || "{'entities': " || '[' || strip(entities) || ']})';

								rc=entity_list.replace(key: tmp_f7, data: tmp_f7, data: entities, data: first_name_list, data: _l);
							 end; 
							 else do;
								tmp_entities='(' || strip(put(position-1,best16.)) || ', ' || strip(put(lengthn(first_name),best16.)) || ", 'PERSON')";
								entities=catx(', ', strip(entities), strip(tmp_entities));
								_l="('" || strip(tmp_f7) || "', " || "{'entities': " || '[' || strip(entities) || ']})';

								if tmp_name_hash.find(key: tmp_f7, key: tmp_entities) = 0 then do;
									rc=entity_list.replace(key: tmp_f7, data: tmp_f7, data: entities, data: first_name_list, data: _l);
								end;  
								else do;
									rc=entity_list.add(key: tmp_f7, data: tmp_f7, data: tmp_entities, data: first_name_list, data: _l);
								end;
							 end;
						  end;
						end;
						else do;
							put "ERROR: F7note: " F7note " has no entity(s): " entities " that is not in the F7note string";
						end;	

			  call prxnext(regex,start, stop,tmp_f7,position,length);
			  end;

			  /* reset CALL PRXNEXT start parameter */
			  start=1;
		%end;

		if last then rc=entity_list.output(dataset:'f7_shredded_2(compress=yes)');
	run;

	/* reorder variables */
	proc sql noprint;
		create table train_data_2 as
		select strip(f) as Name_List label="Names extracted from F7 notes", 
			   strip(_l) as Line label="Labeled F7 notes", 
			   strip(s) as pos label="Entity Positions",
			   strip(k) as F7Note label="Original F7 note"
		from f7_shredded;
	quit;

	/* conditionally export to permanent storage */
	%if %eval(&aaa=1) %then %do;
		libname a "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data";
		proc copy in=work out=a;
			select f7notes f7_shredded train_data;
		run;
		libname a clear;
	%end;
%mend; %string_shredder()

filename a "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\GitRepository\Code\NLP\train_model-6.py";

data _null_;
	file a;
	if _n_=1 then do;
		put 'def find_pii(name):';
		put @5 'return [';
	end;

	%do i=1 %to 26;

		%let qwerty=%scan(&alphabet,&i);
		%let tmp="/&&name_&qwerty/i";

		put @10 "[{'LOWER': {'REGEX': f'" &tmp "'}}],";

		%if %eval(&i=26) %then %do;
			put @5 ']'
		%end;

	%end;

run;



data _null_;
	set train_data end=last;
	file a;

	if _n_=1 then do;
		put "TRAIN_DATA = [";
	end;

	if not last then put @5 line +(-1) ',';

	else if last then do;
		put @5 line;
		put ']'; /* end TRAIN_DATA */
		put ' ';
		put ' ';

		********************************************
		*  --- WRITE spaCy BUILD MODEL SCRIPT  --- *	
		********************************************
		 
		"Intro to NLP with spaCy (4):"
			Detecting programming languages | Episode 4: Named Entity Recognition
			https://youtu.be/IqOJU1-_Fi0

		**************************************************************************;

		/* TRAINING LOOP */
		put "'''";
		put "Intro to NLP with spaCy (4): ";
		put "    Detecting programming languages | Episode 4: Named Entity Recognition";
		put "    https://youtu.be/IqOJU1-_Fi0";
		put "'''";
		put ' ';
		put "from spacy import displacy";
		put "import spacy";
		put "from pathlib import Path";
		put "import random";
		put "import datetime as dt";
		put ' ';
		put "def create_blank_nlp(train_data):";
		put '    nlp = spacy.blank("en")';
		put '    ner = nlp.create_pipe("ner")';
		put "    nlp.add_pipe(ner, last=True)";
		put '    ner = nlp.get_pipe("ner")';
		put "    for _, annotations in train_data:";
		put '        for ent in annotations.get("entities"):';
		put "            ner.add_label(ent[2])";
		put "    return nlp";
		put ' ';
		put "nlp = create_blank_nlp(TRAIN_DATA)";
		put "optimizer = nlp.begin_training()";
		put "for i in range(20):";
		put "    random.shuffle(TRAIN_DATA)";
		put "    losses = {}";
		put "    for text, annotations in TRAIN_DATA:";
		put "        nlp.update([text], [annotations], sgd=optimizer, losses=losses)";
		put '    print(f"Losses at iteration {i} - {dt.datetime.now()}", losses)';
	end;
run;
