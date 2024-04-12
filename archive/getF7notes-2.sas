/* getF7notes.sas
   
   Description: ELT F7 notes from Preview level 3
   Author: Alan Hamm(pqn7@cdc.gov)
   Date: June 2021
*/


* modify as needed;
%let year=2021;
%let month=m03;


*************************** DO NOT EDIT BELOW *************************************;
libname o "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\&year\&month\Preview\level3" access=readonly;

* write text file for NLP;
filename _out "\\cdc.gov\csp_Project\CIPSEA_DHIS_NHIS\Production\NLP_DEV\development_data\f7notes_&month..&year..txt";
data _null_;

	if _n_=1 then do;
		reg=prxparse('s/[[:^ascii:]]//');
		retain reg;
	end;

	length tmp $ 5000;

	file _out;
	infile _in dsd dlm='\n';

	input tmp;

	if anyalpha(tmp)=0 then delete;

	* remove non-ASCII characters;
	call prxchange(reg,-1,tmp);
	
	* remove double quotation;
	tmp=translate(tmp,trimn(''),'"');

	* write to txt file;
	put tmp;
run;

