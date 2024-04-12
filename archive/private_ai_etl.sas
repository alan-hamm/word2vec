/* contemporaneous program to extract data for private-ai python script */
/* author: alan hamm */

libname a "\\cdc.gov\CSP_Private\M728\pqn7\NLP\poc_f7note_files";
libname b "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\2022\M08\level3" access=readonly;

filename c "\\cdc.gov\CSP_Private\M728\pqn7\NLP\poc_f7note_files\f7_to_array.txt";

data a.private_ai2(keep=hhldid f7note);
	set b.m08_f7note;
	if _n_ <= 100;
run;

data _null_;
	length _t $ 512;
	set b.m08_f7note end = last;
	file c;
	if _n_=1 then do;
		put 'f7notes=[';
	end;

	if not last then do;
		_t=cats('"',  strip(f7note), '", ');
		put @5 _t;
	end;

	if last then do;
		_t=cats('"', strip(f7note), '"]');
		put @5 _t;
	end;
	call missing(_t);
run;
