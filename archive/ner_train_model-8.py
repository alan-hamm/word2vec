TRAIN_DATA = [
    ("SHE HAS INSURANCE THRU HER PREVIOUS EMPLOYER AND KELSEY CARE ADVANTAGE W/MEDICARE RX.", {'entities': [(49, 6, 'PERSON')]}),
    ("THIS CAROLINE'S MIDDLE NAME IS MAGDALENE", {'entities': [(5, 8, 'PERSON')]}),
    ("ASSESSOR CLERK: KERRI ANN. TOWN HALL, ASSESSOR OFFICE, 60 CENTER SQUARE, E LONGMEADOW, MA 01028", {'entities': [(16, 5, 'PERSON'),(22, 3, 'PERSON')]}),
    ("INCOMING TELEPHONE CALL FOR RESPONDENT'S DAUGHTER, STEPHANIE DIAZ WHO STATED SHE RECEIVED LETTER TODAY (5/13/21).", {'entities': [(51, 9, 'PERSON'),(61, 4, 'PERSON')]}),
    ("JOSHUA HAS JOINT CUTODY OF HIS TWO CHILDREN WITH HIS EX WIFE, BUT THEY STAY PRIMERIALY WITH THEIR MOTHER", {'entities': [(0, 6, 'PERSON')]}),
    ("HE DOES NOT AHVE PHONE TGIS IS HIS FRIENDS PHONE GLENN SPECK", {'entities': [(49, 5, 'PERSON'),(55, 5, 'PERSON')]}),
    ("ELIANA IS CORRECT SPELLING FOR MRS ROSENWASSER.", {'entities': [(0, 6, 'PERSON'),(35, 11, 'PERSON')]}),
    ("FASTDATA/MARC: 45 FORAKER ST,  EDWARD WALKER,47, TEL: 937-723-9781", {'entities': [(31, 6, 'PERSON'), (38, 6, 'PERSON')]}),
    ("THIS IS AKEEM PHONE NUMBER, HE WILL ASK HEATHER TO CALL ME BACK.", {'entities': [(8, 5, 'PERSON'),(40, 7, 'PERSON')]}),
    ("JAMIE IS TWO SEMESTERS FROM COMPLETING BACHELORS", {'entities': [(0, 5, 'PERSON')]}),
    ("FASTDATA/MARC: 400 WEST FONDERBURG ROAD, APT M, FAIRBORN, LINDA REINHART, NO TELEPHONEFOUND FOR HER NOR HER AGE, WILL FOLLOW WITH ANOTHER SEARCH ENGINE", {'entities': [(58, 5, 'PERSON'),(66, 8, 'PERSON')]}),
    ("FRIDAY @ 9AM 562 324 5299 MICHELLE", {'entities': [(26, 8, 'PERSON')]}),
    ("LYNDA STATED THAT SHE DOES GO TO ALL OF HIS DOCTORS TO HELP HIM REMEMBER", {'entities': [(0, 5, 'PERSON')]}),
    ("MS. LORETTA REFUSES-HE CALLED WHILE SHE WAS OUT OF THE HOUSE-HE WAS WILLING-SHE WILL NOT DO IT", {'entities': [(4, 7, 'PERSON')]}),
    ("DIANE AND JARON ARE BRANDIE'S NEICE AND NEPHEW SANNTONIO AND ZMARI ARE MARCUS GRANDCHILDREN", {'entities': [(0, 5, 'PERSON'),(20, 7, 'PERSON'), (47, 9, 'PERSON'), (61, 5, 'PERSON'), (71, 6, 'PERSON')]}),
    ("DAD STOPPED DRINKING WHEN JESUS WAS SIX", {'entities': [(26, 5, 'PERSON')]}),
    ("20,000 IS MARGARITA", {'entities': [(10, 9, 'PERSON')]}),
    ("PROXY HAD TO LEAVE - TO TAKE THOMAS TO A DR'S APPT.", {'entities': [(29, 6, 'PERSON')]}),
    ("CAROLYN HAS BEEN THERE FOR 3 YEARS", {'entities': [(0, 7, 'PERSON')]}),
    ("MIRANDA IS ANSWERING/USED AS PROXY B/C ELIZABETH HAS NOT SLEPT IN 3 DAYS DUE TO ILLNES", {'entities': [(0, 7, 'PERSON'),(39, 9, 'PERSON')]}),
    ("CAME IN DEC 2016 FROM COLORADO TO LIVE WITH SISTER, BERNICE", {'entities': [(52, 7, 'PERSON')]}),
    ("MICHAEL HAS LIVED HERE FOR 4 YEARS", {'entities': [(0, 7, 'PERSON')]}),
    ("DOCTOR NEVER TOLD VANESSA SHE HAD COVID !9.", {'entities': [(18, 7, 'PERSON')]}),
    ("I STARTED OFF THE SURVEY WITH MATTHEW NISSEN HE ANSWERED THE FIRST PART AND THEN SAMPLE CHILD HIS WIFE WAS CHOSEN AS SAMPLE ADULT AND WHEN HE GOT HOME HE HANDED HIS PHNE OFF TO HIS WIFE SHE FINISHED", {'entities': [(30, 7, 'PERSON'),(38, 6, 'PERSON')]}),
    ("FASTDATA/MARC,  61 FORAKER ST, XENIA, NICOLAS LINGE,38, NO PHONE NUMBER FOUND", {'entities': [(31, 5, 'PERSON'),(38, 7, 'PERSON'),(64, 5, 'PERSON')]}),
    ("RAYNE LIBRARY 10AM STEVEN BORNE.CATHOLIC CHURCH CUTTING GRASS PERRIDON.WEDNESDAY.", {'entities': [(19, 6, 'PERSON'),(26, 5, 'PERSON')]}),
    ("NAME IS BYRON", {'entities': [(8, 5, 'PERSON')]}),
    ("MARIANNA ALLEN LOH SAID THEY ARE DECLINING TO VOLUNTEER AT THIS TIME.S", {'entities': [(0, 8, 'PERSON'),(9, 5, 'PERSON'),(15, 3, 'PERSON')]}),
    ("YES FROM WIFE BIANCA WITH LEG & COVID 19.", {'entities': [(14, 6, 'PERSON')]}),
    ("PAULINE HAS HER OWN BCBS. RICHARD HAS HIS OWN BCBS WITH MOLLY.", {'entities': [(56, 5, 'PERSON'), (0, 7, 'PERSON'), (26, 7, 'PERSON')]}),
    ("JORDAN NO LONGER LIVES HERE, MOVED TO CHICAGO.", {'entities': [(0, 6, 'PERSON')]}),
    ("MOM CLAUDIA IS 38 AND JUAN DAD IS 40", {'entities': [(4, 7, 'PERSON'),(22, 4, 'PERSON')]}),
    ("TWO OF ROSTER HAVE THE SAME NAME.  CLAUDIA-MOM IS 38 AND JUAN  DAD IS 40", {'entities': [(35, 7, 'PERSON'),(57, 4, 'PERSON')]}),
    ("ONLY VERONICA AND HER MOTHER", {'entities': [(5, 8, 'PERSON')]}),
    ("MOM CARRIE DOES NOT APPROVE ANY ONE TO DISTURB HER AT SCHOOL  MOM IS HAPPY TO BE THE PROXY.", {'entities': [(4, 6, 'PERSON')]}),
    ("MICHAEL IS A WHEELCHAIR FOR FOUR MONTH TO TAKE PRESSURE OFF OF HIS HIP JOINT, HE DOES WALK AT TIMES.", {'entities': [(0, 7, 'PERSON')]}),
    ("VERNON IS ON DISABILITY/SSI, SUSAN IS RETIRED AND RECEIVES FOOD STAMPS", {'entities': [(29, 5, 'PERSON'), (0, 6, 'PERSON')]}),
    ("MELISSA REFUSED TO PARTICIPATE", {'entities': [(0, 7, 'PERSON')]}),
    ("THIS IS ONLY BARBARA'S INCOME", {'entities': [(13, 7, 'PERSON')]}),
    ("HE ISN'TSURE OF ELENA'S, ESTIMATED HER PORTION", {'entities': [(16, 5, 'PERSON')]}),
    ("FASTDATA/MARC: DENIS DZIECH,57, 513-625-2302", {'entities': [(15, 5, 'PERSON'),(21, 6, 'PERSON')]}),
    ("JORDAN HAD THE OPPORTUNITY, BUT SHE DECLINED", {'entities': [(0, 6, 'PERSON')]}),
    ("I AM SPEAKING TO CAROL.  SHE IS CARRIE'S MOTHER AND LEGAL GUARDIAN.   SHE TOLD ME THAT CARRIE IS VERBAL BUT HAS SEVER MENTAL HEALTH ISSUES THAT WOULD PREVENT HER FROM ANSWERING THE QUESTIONS.   I AM", {'entities': [(17, 5, 'PERSON'), (32, 6, 'PERSON'), (87, 6, 'PERSON')]}),
    ("JAMES DIDN'T WANT TO RECEIVE ANY PHONE CALL.", {'entities': [(0, 5, 'PERSON')]}),
    ("DANNY AND LADONNA ARE MARRIED AND  BOTH DISABLED WITH LIMITED FUNDS.  THEY LIVE IN THIS HU WITH DANIELLE, THEIR DAUGHTER, AND HER FAMILY.  DANIELLE AND JUSTIN WORK AND DAN HANDLES KIDS.  DAN DOES NO", {'entities': [(0, 5, 'PERSON'), (96, 8, 'PERSON'), (139, 8, 'PERSON'), (152, 6, 'PERSON'), (10, 7, 'PERSON')]}),
    ("STEPHANIE FIGUEROA/JUAN O'NEILL", {'entities': [(0, 9, 'PERSON'),(10, 8, 'PERSON'),(19, 4, 'PERSON'),(24, 6, 'PERSON')]}),
    ("4573 IS MARGARITA", {'entities': [(8, 9, 'PERSON')]}),
    ("SPOKE WITH FELICIA BARNES WHO LIVES IN THE APARTMENT BELOW 28, STATED NO ONE LIVES IN THAT APARTMENT AT THIS TIME.  318 237 0631 IS HER PHONE NUMBER.", {'entities': [(11, 7, 'PERSON'),(19, 6, 'PERSON')]}),
    ("BROOKE@CAPSTONEEMULTIFAMILY.COM", {'entities': [(0, 6, 'PERSON')]}),
    ("TWO OF ROSTER HAVE THE SAME NAME.  CLAUDIA-MOM IS 38", {'entities': [(35, 7, 'PERSON')]}),
    ("270-2277874 BETTY PETERS", {'entities': [(12, 5, 'PERSON'),(18, 6, 'PERSON')]}),
    ("VICTOR'S FRIEND IS TRANSLATING INTO SPANISH FOR THE INTERVIEW.", {'entities': [(0, 6, 'PERSON')]}),
    ("THOMAS HAS CANCER IN HIS FACE AND JAW. HE HAS A GREAT DEAL OF TROUBLE SPEAKING.", {'entities': [(0, 6, 'PERSON')]}),
    ("JUST HER AND DON. SHE DOES NOT KNOW HOW MUCH MICHAEL MAKES.", {'entities': [(13, 3, 'PERSON'),(45, 7, 'PERSON')]}),
    ("MARGARET FELL ON EASTER SUNDAY AND MAY HAVE A FRACTURE IN HER ANKLE, SO THE PAIN IS DUE TO THE FALL NOT AS A CHRONIC CONDITION", {'entities': [(0, 8, 'PERSON')]}),
    ("MS. CAROL STATES THAT AFTER PAYING HER BILLS SHE DOESN'T HAVE ENOUGH MONEY TO PAY FOR HER DENTAL AND MEDICAID WILL ONLY PAY FOR THEM TO BE PULLED; NOT CLEANED OR CAPS, ETC.", {'entities': [(4, 5, 'PERSON')]}),
    ("THE SAMPLE ADULT INTERVIEW BY PERSONAL VISIT AND CHILD SAMPLE INTERVIEW BY TELEPHONE DUE TO LESLIE DIDN'T HAVE TIME TO COMPLETE THE INTERVIEW AT ONE TIME.", {'entities': [(92, 6, 'PERSON')]}),
    ("FASTDATA/MARC/ 6599 GOSHEN RD, GOSHEN, LOUIS CHILBS, 51", {'entities': [(39, 5, 'PERSON'),(45, 6, 'PERSON')]}),
    ("HALEY I BLACK AND LATINA", {'entities': [(0, 5, 'PERSON')]}),
    ("MICHAEL DOES NOT KNOW ABOUT THIS TYPE OF CARE AS IT PRIOR TO WHEN HE STARTED TAKING CARE OF HER", {'entities': [(0, 7, 'PERSON')]}),
    ("NATALIE WORKS ODD HOURS AT A RESTAURANT & WILL B GOING TO THE SHORE FOR MEMORIAL DAY WKND' SO I DON'T KNOW IF I WILL B HEARING FROM HER", {'entities': [(0, 7, 'PERSON')]}),
    ("CASE 3552, 636 FAIRFIELD AVE,FAIRBORN, JONATHAN EDWIN WARD, 27 YEARS OLD ,937,624,2875", {'entities': [(48, 5, 'PERSON'), (39, 8, 'PERSON'),(56, 4, 'PERSON')]}),
    ("CATHY", {'entities': [(0, 5, 'PERSON')]}),
    ("FASTDATA/MARC,1166 JAPER AVE,HAROLD MATHSON,57, NO PHONE NUMBER FOR THIS RESIDENCE, WILL TRY A DIFFERENT SOURCE", {'entities': [(29, 6, 'PERSON'),(36, 7, 'PERSON')]}),
    ("BRYON TRAVELS A LOT, BUT STAYS HERE MORE THAN HALF THE NIGHTS IN A YEAR", {'entities': [(0, 5, 'PERSON')]}),
    ("HE GOES BY SCOTT.", {'entities': [(11, 5, 'PERSON')]}),
    ("LYNDA STATED THAT SHE DOES GO TO ALL OF HIS DOCTORS TO HELP HIM REMEMBER - HE IS NOT CLAIMING THAT", {'entities': [(0, 5, 'PERSON')]}),
    ("PATTY HAS DEMENTIA & WAS IN A NURSING HOME IN IDAHO AT START OF PANDEMIC, THEY BROUGHT HER HOME WHEN THERE WAS AN OUTBREAK & BEFORE SHE GOT IT, BUT SHE DOESN'T HAVE ANY AWARENESS OF IT REALLY -", {'entities': [(0, 5, 'PERSON')]}),
    ("MS. CAROL ALSO HAS TO PAY FOR HER GLASSES OUT OF POCKET.  MEDICAID WILL ONLY PAY FOR THE EXAMS.", {'entities': [(4, 5, 'PERSON')]}),
    ("PV. SERGEY KORKIN ANSWERED THE DOOR.  HE SAID HE READ THE ADV. LETTER I HAD LEFT YESTERDAY AND THAT HE IS REFUSING TO PARTICIPATE.  THE TONE OF HIS VOICE AND THE LOOK HE GAVE ME WAS VERY UNCOMFORTABL", {'entities': [(4, 6, 'PERSON'),(11, 6, 'PERSON')]})
]
 
 
'''
Intro to NLP with spaCy (4): 
    Detecting programming languages | Episode 4: Named Entity Recognition
    https://youtu.be/IqOJU1-_Fi0
 2021 MONTHS APRIL AND MAY
'''
 
from spacy import displacy
import spacy
from spacy.util import minibatch, compounding
import en_core_web_lg
import datetime as dt
 
def create_blank_nlp(train_data):
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, first=True)
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    return nlp
 
nlp = create_blank_nlp(TRAIN_DATA)
optimizer = nlp.begin_training()
for i in range(25):
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(2.0,32.0,1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(
			texts, # batch of texts
			annotations, # batches of annotations
			drop=0.1, # dropout - make it harder to mermorize data
			losses=losses
		)
    print(f"Lossess at iteration {i} - {dt.datetime.now()} {losses}")
 
nlp.to_disk("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\model")


f=open('\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\train_data.txt','r').readlines()
with open('\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\train_data.txt','r') as f:
    lines = f.readlines()
f.close() 

f7lines=''.join(lines)
doc=nlp(f7lines)

html = displacy.render([doc], style="ent")
file = open("\\\\cdc.gov\\csp_Project\\CIPSEA_DHIS_NHIS\\Production\\NLP_DEV\\development_data\\train_data.html","w")
file.write(html)
file.write(html)
file.close()