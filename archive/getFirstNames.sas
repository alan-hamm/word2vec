*
	author alan hamm(pqn7@cdc.gov)
	date august 2021
	description - read and clean first names worksheet output csv file
*;

proc import datafile="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\GitRepository\Reference\NLP\firstnames.xlsx"
	dbms=excel
	out=firstnames
	replace;
run;

libname b "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\GitRepository\Reference\NLP";

data b.firstnames_complete;
	set firstnames;
run;

data b.firstnames_cleaned;
	set firstnames(keep=firstname);

	firstname=transtrn(firstname,'"',trimn(''));
	firstname=strip(firstname);
	firstname = translate(firstname,'','0A0D'x);
run;

proc export data=b.firstnames_cleaned
	outfile="\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\production_data\first_names.csv"
	dbms=csv
	replace;
run;
