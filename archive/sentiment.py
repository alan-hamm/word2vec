"""
author alan hamm(pqn7)
date september 2022

script to perform sentiment analysis of PII_Master case notes and
contact history instrument
"""

import pandas as pd
import matplotlib.pyplot as plt
import pprint

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import seaborn as sns

from nltk.tokenize import RegexpTokenizer, word_tokenize
regexp=RegexpTokenizer('\w+')


def create_dataframe(data_frame, var, datapath):
    data_frame = pd.DataFrame(columns=['OUTCOME','YEAR','QUARTER', 'MONTH', var])
    var = pd.read_sas(datapath, 
                     encoding='latin-1')
    var = var[['OUTCOME', 'year','quarter','month', 'CTOTHER']]
    #print(df.head())
    #reassignment and datatype defensive programming
    for row in var:
        data_frame['OUTCOME']    = var['OUTCOME']
        data_frame['YEAR']       = var['year']
        data_frame['QUARTER']    = var['quarter']
        data_frame['MONTH']      = var['month']
        data_frame[var]      = var[var.to_string()]
    return data_frame

""" ETL DATA """
CTOTHER=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_CTOTHER.sas7bdat", encoding='latin-1')
pprint.pprint(CTOTHER.head(n=5))

PNONCONOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_PNONCONOTH.sas7bdat" ,encoding='latin-1')
pprint.pprint(PNONCONOTH.head(n=5))

PSPECLANG=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_PSPECLANG.sas7bdat",
                      encoding='latin-1')
pprint.pprint(CTOTHER.head(n=5))

PSTRATOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_PSTRATOTH.sas7bdat",
                      encoding='latin-1')
pprint.pprint(PSTRATOTH.head(n=5))

PRSPNDOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_PRSPNDOTH.sas7bdat",
                      encoding='latin-1')
pprint.pprint(PRSPNDOTH.head(n=5))

RSPNTOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_RSPNTOTH.sas7bdat",
                     encoding='latin-1')
pprint.pprint(PRSPNDOTH.head(n=5))

SPECNOATTEMPT=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_SPECNOATTEMPT.sas7bdat",
                          encoding='latin-1')
pprint.pprint(SPECNOATTEMPT.head(n=5))

NCTPEROT=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_NCTPEROT.sas7bdat",
                     encoding='latin-1')
pprint.pprint(NCTPEROT.head(n=5))

NCTTELOT=pd.read_sas( r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_NCTTELOT.sas7bdat",
                     encoding='latin-1')
pprint.pprint(NCTTELOT.head(n=5))

STRATOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\Testing\PQN7nlp\corpora\sentiment_analysis\adam_STRATOTH.sas7bdat",
                     encoding='latin-1')
pprint.pprint(STRATOTH.head(n=5))


""" tokenize text fields """
CTOTHER['TEXT_TOKEN'] = CTOTHER['CTOTHER'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
PSPECLANG['TEXT_TOKEN'] = PSPECLANG['PSPECLANG'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
PNONCONOTH['TEXT_TOKEN'] = PNONCONOTH['PNONCONOTH'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
PSTRATOTH['TEXT_TOKEN'] = PSTRATOTH['PSTRATOTH'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
PRSPNDOTH['TEXT_TOKEN'] = PRSPNDOTH['PRSPNDOTH'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
RSPNTOTH['TEXT_TOKEN'] = RSPNTOTH['RSPNTOTH'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
SPECNOATTEMPT['TEXT_TOKEN'] = SPECNOATTEMPT['SPECNOATTEMPT'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
NCTPEROT['TEXT_TOKEN'] = NCTPEROT['NCTPEROT'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
NCTTELOT['TEXT_TOKEN'] = NCTTELOT['NCTTELOT'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)
STRATOTH['TEXT_TOKEN'] = STRATOTH['STRATOTH'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)


""" remove stop words """
stop_words = stopwords.words('english')
CTOTHER['TEXT_TOKEN']=CTOTHER['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
PSPECLANG['TEXT_TOKEN'] = PSPECLANG['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
PNONCONOTH['TEXT_TOKEN'] = PNONCONOTH['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
PSTRATOTH['TEXT_TOKEN'] = PSTRATOTH['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
RSPNTOTH['TEXT_TOKEN'] = RSPNTOTH['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
PRSPNDOTH['TEXT_TOKEN'] = PRSPNDOTH['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
SPECNOATTEMPT['TEXT_TOKEN'] = SPECNOATTEMPT['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
NCTPEROT['TEXT_TOKEN'] = NCTPEROT['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
NCTTELOT['TEXT_TOKEN'] = NCTTELOT['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])


""" combine tokens into a list """
CTOTHER['text_string'] = CTOTHER['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
PSPECLANG['text_string'] = PSPECLANG['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
PNONCONOTH['text_string'] = PNONCONOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
PSTRATOTH['text_string'] = PSTRATOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
RSPNTOTH['text_string'] = RSPNTOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
PRSPNDOTH['text_string'] = PRSPNDOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
SPECNOATTEMPT['text_string'] = SPECNOATTEMPT['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
NCTPEROT['text_string'] = NCTPEROT['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
NCTTELOT['text_string'] = NCTTELOT['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))


""" get all words in text_string """
all_wordsCTOTHER = ' '.join([word for word in CTOTHER['text_string']])
all_wordsPSPECLANG = ' '.join([word for word in PSPECLANG['text_string']])
all_wordsPNONCONOTH = ' '.join([word for word in PNONCONOTH['text_string']])
all_wordsPSTRATOTH = ' '.join([word for word in PSTRATOTH['text_string']])
all_wordsRSPNTOTH = ' '.join([word for word in RSPNTOTH['text_string']])
all_wordsPRSPNDOTH = ' '.join([word for word in PRSPNDOTH['text_string']])
all_wordsSPECNOATTEMPT = ' '.join([word for word in SPECNOATTEMPT['text_string']])
all_wordsNCTPEROT = ' '.join([word for word in NCTPEROT['text_string']])
all_wordsNCTTELOT = ' '.join([word for word in NCTTELOT['text_string']])


""" tokenize with word_tokenize """
tokenized_CTOTHER = word_tokenize(all_wordsCTOTHER)
tokenized_PSPECLANG = word_tokenize(all_wordsPSPECLANG)
tokenized_PNONCONOTH = word_tokenize(all_wordsPNONCONOTH)
tokenized_PSTRATOTH = word_tokenize(all_wordsPSTRATOTH)
tokenized_RSPNTOTH = word_tokenize(all_wordsRSPNTOTH)
tokenized_PRSPNDOTH = word_tokenize(all_wordsPRSPNDOTH)
tokenized_SPECNOATTEMPT = word_tokenize(all_wordsSPECNOATTEMPT)
tokenized_NCTPEROT = word_tokenize(all_wordsNCTPEROT)
tokenized_NCTTELOT = word_tokenize(all_wordsNCTTELOT)


""" derive frequency distribution of words """
fdistCTOTHER = FreqDist(tokenized_CTOTHER)
fdistPSPECLANG = FreqDist(tokenized_PSPECLANG)
fdistPNONCONOTH = FreqDist(tokenized_PNONCONOTH)
fdistPSTRATOTH = FreqDist(tokenized_PSTRATOTH)
fdistRSPNTOTH = FreqDist(tokenized_RSPNTOTH)
fdistPRSPNDOTH = FreqDist(tokenized_PRSPNDOTH)
fdistSPECNOATTEMPT = FreqDist(tokenized_SPECNOATTEMPT)
fdistNCTPEROT = FreqDist(tokenized_NCTPEROT)
fdistNCTTELOT = FreqDist(tokenized_NCTTELOT)
#pprint.pprint(fdist)
#plt.figure(figsize=(175,10))
#plt.xticks(fontsize=14, rotation=45)
#fdist.plot()


""" assign values based on frequency distribution where word count is greater than 2 """
CTOTHER['text_string_fdist'] = CTOTHER['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistCTOTHER[item] >= 1]))
PSPECLANG['text_string_fdist'] = PSPECLANG['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistPSPECLANG[item] >= 1]))
PNONCONOTH['text_string_fdist'] = PNONCONOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistPNONCONOTH[item] >= 1]))
PSTRATOTH['text_string_fdist'] = PSTRATOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistPSTRATOTH[item] >= 1]))
RSPNTOTH['text_string_fdist'] = RSPNTOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistRSPNTOTH[item] >= 1]))
PRSPNDOTH['text_string_fdist'] = PRSPNDOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistPRSPNDOTH[item] >= 1]))
SPECNOATTEMPT['text_string_fdist'] = SPECNOATTEMPT['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistSPECNOATTEMPT[item] >= 1]))
NCTPEROT['text_string_fdist'] = NCTPEROT['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistNCTPEROT[item] >= 1]))
NCTTELOT['text_string_fdist'] = NCTTELOT['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistNCTTELOT[item] >= 1]))


""" lemmatize text """
wordnet_lem = WordNetLemmatizer()
CTOTHER['text_string_lem'] = CTOTHER['text_string_fdist'].apply(wordnet_lem.lemmatize)
PSPECLANG['text_string_lem'] = PSPECLANG['text_string_fdist'].apply(wordnet_lem.lemmatize)
PNONCONOTH['text_string_lem'] = PNONCONOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
PSTRATOTH['text_string_lem'] = PSTRATOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
RSPNTOTH['text_string_lem'] = RSPNTOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
PRSPNDOTH['text_string_lem'] = PRSPNDOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
SPECNOATTEMPT['text_string_lem'] = SPECNOATTEMPT['text_string_fdist'].apply(wordnet_lem.lemmatize)
NCTPEROT['text_string_lem'] = NCTPEROT['text_string_fdist'].apply(wordnet_lem.lemmatize)
NCTTELOT['text_string_lem'] = NCTTELOT['text_string_fdist'].apply(wordnet_lem.lemmatize)

analyzer = SentimentIntensityAnalyzer()
CTOTHER['polarity'] = CTOTHER["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
PSPECLANG['polarity'] = PSPECLANG["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
PNONCONOTH['polarity'] = PNONCONOTH["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
PSTRATOTH['polarity'] = PSTRATOTH["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
RSPNTOTH['polarity'] = RSPNTOTH["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
PRSPNDOTH['polarity'] = PRSPNDOTH["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
SPECNOATTEMPT['polarity'] = SPECNOATTEMPT["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
NCTPEROT['polarity'] = NCTPEROT["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
NCTTELOT['polarity'] = NCTTELOT["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))


CTOTHER = pd.concat([ CTOTHER.drop(['VISITDATE'],axis=1), CTOTHER['polarity'].apply(pd.Series)], axis=1)
PSPECLANG = pd.concat([ PSPECLANG.drop(['VISITDATE'],axis=1), PSPECLANG['polarity'].apply(pd.Series)], axis=1)
PNONCONOTH = pd.concat([PNONCONOTH.drop(['VISITDATE'],axis=1), PNONCONOTH['polarity'].apply(pd.Series)], axis=1)
PSTRATOTH = pd.concat([ PSTRATOTH.drop(['VISITDATE'],axis=1), PSTRATOTH['polarity'].apply(pd.Series)], axis=1)
RSPNTOTH = pd.concat([ RSPNTOTH.drop(['VISITDATE'],axis=1), RSPNTOTH['polarity'].apply(pd.Series)], axis=1)
PRSPNDOTH = pd.concat([ PRSPNDOTH.drop(['VISITDATE'],axis=1), PRSPNDOTH['polarity'].apply(pd.Series)], axis=1)
SPECNOATTEMPT = pd.concat([SPECNOATTEMPT.drop(['VISITDATE'],axis=1),  SPECNOATTEMPT['polarity'].apply(pd.Series)], axis=1)
NCTPEROT = pd.concat([NCTPEROT.drop(['VISITDATE'],axis=1), NCTPEROT['polarity'].apply(pd.Series)], axis=1)
NCTTELOT = pd.concat([NCTTELOT.drop(['VISITDATE'],axis=1), NCTTELOT['polarity'].apply(pd.Series)], axis=1)


CTOTHER['sentiment'] = CTOTHER['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
PSPECLANG['sentiment'] = PSPECLANG['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
PNONCONOTH['sentiment'] = PNONCONOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
PSTRATOTH['sentiment'] = PSTRATOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
RSPNTOTH['sentiment'] = RSPNTOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
PRSPNDOTH['sentiment'] = PRSPNDOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
SPECNOATTEMPT['sentiment'] = SPECNOATTEMPT['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
NCTPEROT['sentiment'] = NCTPEROT['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
NCTTELOT['sentiment'] = NCTTELOT['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')


"""
plt.figure(figsize=(8,5))
plt.subplot()
sns.countplot(y='sentiment',
              data=df,
              palette=['#b2d8d8', '#008080', '#db3d13'])
"""  


""" CTOTHER SENTIMENT ONLY """
fig, axes = plt.subplots(1,3, figsize=(15,5))
plt.suptitle('CTOTHER')
axes[0].plot(CTOTHER['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(CTOTHER['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(CTOTHER['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

""" CTOTHER MONTH """
plt.figure(figsize=(8,5))              
j=sns.catplot(data=CTOTHER, x='month', y='neg', col='OUTCOME')
plt.legend(bbox_to_anchor=(1.5,1), borderaxespad=0, loc='lower left', labels=['neutral', 'positive', 'negative']) 

plt.figure(figsize=(8,5))                
j=sns.catplot(data=CTOTHER, x='month', y='pos', col='OUTCOME')
plt.legend(bbox_to_anchor=(1.5,1), borderaxespad=0, loc='lower left', labels=['neutral', 'positive', 'negative']) 

plt.figure(figsize=(8,5))                
j=sns.catplot(data=CTOTHER, x='month', y='compound', col='OUTCOME')
plt.legend(bbox_to_anchor=(1.5,1), borderaxespad=0, loc='lower left', labels=['neutral', 'positive', 'negative']) 

""" CTOTHER QUARTER """
plt.figure(figsize=(8,5))                
j=sns.catplot(data=CTOTHER, x='quarter', y='neg', col='OUTCOME')
plt.legend(bbox_to_anchor=(1.5,1), borderaxespad=0, loc='lower left', labels=['neutral', 'positive', 'negative']) 

plt.figure(figsize=(8,5))                
j=sns.catplot(data=CTOTHER, x='quarter', y='pos', col='OUTCOME')
plt.legend(bbox_to_anchor=(1.5,1), borderaxespad=0, loc='lower left', labels=['neutral', 'positive', 'negative']) 

plt.figure(figsize=(8,5))                
j=sns.catplot(data=CTOTHER, x='quarter', y='compound', col='OUTCOME')
plt.legend(bbox_to_anchor=(1.5,1), borderaxespad=0, loc='lower left', labels=['neutral', 'positive', 'negative']) 




""" PSTRATOTH """
plt.figure(figsize=(8,5))                
j=sns.catplot(data=PSTRATOTH, x='month', y='neg', col='OUTCOME')
plt.legend(bbox_to_anchor=(1.5,1), borderaxespad=0, loc='lower left', labels=['neutral', 'positive', 'negative']) 


fig, axes = plt.subplots(1,3, figsize=(10,5))
plt.suptitle('PSTRATOTH')
axes[0].plot(PSTRATOTH['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(PSTRATOTH['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(PSTRATOTH['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

""" RSPNTOTH """
fig, axes = plt.subplots(1,3, figsize=(15,10))
plt.suptitle('RSPNTOTH')
axes[0].plot(RSPNTOTH['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(RSPNTOTH['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(RSPNTOTH['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

""" PRSPNDOTH """
fig, axes = plt.subplots(1,3, figsize=(15,10))
plt.suptitle('PRSPNDOTH')
axes[0].plot(PRSPNDOTH['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(PRSPNDOTH['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(PRSPNDOTH['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

""" SPECNOATTEMPT """
fig, axes = plt.subplots(1,3, figsize=(15,10))
plt.suptitle('SPECNOATTEMPT')
axes[0].plot(SPECNOATTEMPT['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(SPECNOATTEMPT['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(SPECNOATTEMPT['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

""" NCTPEROT """
fig, axes = plt.subplots(1,3, figsize=(15,10))
plt.suptitle('NCTPEROT')
axes[0].plot(NCTPEROT['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(NCTPEROT['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(NCTPEROT['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

""" NCTTELOT """
fig, axes = plt.subplots(1,3, figsize=(15,10))
plt.suptitle('NCTTELOT')
axes[0].plot(NCTTELOT['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(NCTTELOT['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(NCTTELOT['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

""" PSPECLANG """
fig, axes = plt.subplots(1,3, figsize=(10,5))
plt.suptitle('PSPECLANG')
axes[0].plot(PSPECLANG['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(PSPECLANG['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(PSPECLANG['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

""" PNONCONOTH """
fig, axes = plt.subplots(1,3, figsize=(10,5))
plt.suptitle('PNONCONOTH')
axes[0].plot(PNONCONOTH['neg'], color='blue')
axes[0].set_xlabel('Negative Sentiment')
axes[1].plot(PNONCONOTH['pos'], color='red')
axes[1].set_xlabel('Positive Sentiment')
axes[2].plot(PNONCONOTH['compound'], color='green')
axes[2].set_xlabel('Compound Sentiment')

