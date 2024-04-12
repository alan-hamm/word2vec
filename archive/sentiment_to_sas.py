"""
author alan hamm(pqn7)
date september 2022

script to perform sentiment analysis of PII_Master case notes and
contact history instrument
"""

import pandas as pd
import pprint

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import RegexpTokenizer, word_tokenize
regexp=RegexpTokenizer('\w+')


""" ETL DATA """
CTOTHER=pd.read_sas(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_CTOTHER.sas7bdat", 
											encoding='latin-1')
#pprint.pprint(CTOTHER.head(n=5))

PNONCONOTH=pd.read_sas(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_PNONCONOTH.sas7bdat" ,
											encoding='latin-1')
#pprint.pprint(PNONCONOTH.head(n=5))

PSPECLANG=pd.read_sas(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_PSPECLANG.sas7bdat",
                      encoding='latin-1')
#pprint.pprint(CTOTHER.head(n=5))

PSTRATOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_PSTRATOTH.sas7bdat",
                      encoding='latin-1')
#pprint.pprint(PSTRATOTH.head(n=5))

PRSPNDOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_PRSPNDOTH.sas7bdat",
                      encoding='latin-1')
#pprint.pprint(PRSPNDOTH.head(n=5))

RSPNTOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_RSPNTOTH.sas7bdat",
                     encoding='latin-1')
#pprint.pprint(PRSPNDOTH.head(n=5))

SPECNOATTEMPT=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_SPECNOATTEMPT.sas7bdat",
                          encoding='latin-1')
#pprint.pprint(SPECNOATTEMPT.head(n=5))

SPECLANG=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_SPECLANG.sas7bdat",
                          encoding='latin-1')
#pprint.pprint(SPECLANG.head(n=5))

NCTPEROT=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_NCTPEROT.sas7bdat",
                     encoding='latin-1')
#pprint.pprint(NCTPEROT.head(n=5))

NCTTELOT=pd.read_sas( r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_NCTTELOT.sas7bdat",
                     encoding='latin-1')
#pprint.pprint(NCTTELOT.head(n=5))

STRATOTH=pd.read_sas(r"\\cdc.gov\csp_project\CIPSEA_PII_NHIS_EXCHANGE\Census\TextAnalysis\corpus\sentiment_analysis\adam_STRATOTH.sas7bdat",
                     encoding='latin-1')
#pprint.pprint(STRATOTH.head(n=5))


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
#SPECLANG['TEXT_TOKEN'] = SPECLANG['SPECLANG'].replace(r'(\n[0-9]+)', ' ').apply(regexp.tokenize)


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
STRATOTH['TEXT_TOKEN'] = STRATOTH['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])
#SPECLANG['TEXT_TOKEN'] = SPECLANG['TEXT_TOKEN'].apply(lambda x: [item for item in x if item not in stop_words])


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
STRATOTH['text_string'] = STRATOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))
#SPECLANG['text_string'] = SPECLANG['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if len(item) > 2]))


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
all_wordsSTRATOTH = ' '.join([word for word in STRATOTH['text_string']])
#all_wordsSPECLANG = ' '.join([word for word in SPECLANG['text_string']])

""" build wordcloud """
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def gen_word_cloud(variable, text):
    wordcloud = WordCloud(width=600,
                          height=400,
                          random_state=2,
                          max_font_size=100).generate(text)
    plt.figure(figsize=(10,7))
    plt.title(variable)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

gen_word_cloud('CTOTHER', all_wordsCTOTHER)
gen_word_cloud('PSPECLANG', all_wordsPSPECLANG)
gen_word_cloud('PNONCONOTH', all_wordsPNONCONOTH)
gen_word_cloud('PSTRATOTH', all_wordsPSTRATOTH)
gen_word_cloud('RSPNTOTH', all_wordsRSPNTOTH)
gen_word_cloud('PRSPNDOTH', all_wordsPRSPNDOTH)
gen_word_cloud('SPECNOATTEMPT', all_wordsSPECNOATTEMPT)
gen_word_cloud('NCTPEROT', all_wordsNCTPEROT)
gen_word_cloud('NCTTELOT', all_wordsNCTTELOT)
gen_word_cloud('STRATOTH', all_wordsSTRATOTH)


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
tokenized_STRATOTH = word_tokenize(all_wordsSTRATOTH)
#tokenized_SPECLANG = word_tokenize(all_wordsSPECLANG)


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
fdistSTRATOTH = FreqDist(tokenized_STRATOTH)
#fdistSPECLANG = FreqDist(tokenized_SPECLANG)
#pprint.pprint(fdist)
#plt.figure(figsize=(175,10))
#plt.xticks(fontsize=14, rotation=45)
#fdist.plot()

""" MOST COMMON WORDS """
top_10 = fdistCTOTHER.most_common(10)
fdist = pd.Series(dict(top_10))
import seaborn as sns
sns.barplot(y=fdist.index, x=fdist.values, color='blue')

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
STRATOTH['text_string_fdist'] = STRATOTH['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistSTRATOTH[item] >= 1]))
#SPECLANG['text_string_fdist'] = SPECLANG['TEXT_TOKEN'].apply(lambda x: ' '.join([item for item in x if fdistSPECLANG[item] >= 1]))



wordnet_lem=WordNetLemmatizer()
CTOTHER['text_string_lem'] = CTOTHER['text_string_fdist'].apply(wordnet_lem.lemmatize)
PSPECLANG['text_string_lem'] = PSPECLANG['text_string_fdist'].apply(wordnet_lem.lemmatize)
PNONCONOTH['text_string_lem'] = PNONCONOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
PSTRATOTH['text_string_lem'] = PSTRATOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
RSPNTOTH['text_string_lem'] = RSPNTOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
PRSPNDOTH['text_string_lem'] = PRSPNDOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
SPECNOATTEMPT['text_string_lem'] = SPECNOATTEMPT['text_string_fdist'].apply(wordnet_lem.lemmatize)
NCTPEROT['text_string_lem'] = NCTPEROT['text_string_fdist'].apply(wordnet_lem.lemmatize)
NCTTELOT['text_string_lem'] = NCTTELOT['text_string_fdist'].apply(wordnet_lem.lemmatize)
STRATOTH['text_string_lem'] = STRATOTH['text_string_fdist'].apply(wordnet_lem.lemmatize)
#SPECLANG['text_string_lem'] = SPECLANG['text_string_fdist'].apply(wordnet_lem.lemmatize)


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
STRATOTH['polarity'] = STRATOTH["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))
#SPECLANG['polarity'] = SPECLANG["text_string_lem"].apply(lambda x: analyzer.polarity_scores(x))


CTOTHER = pd.concat([ CTOTHER.drop(['VISITDATE'],axis=1), CTOTHER['polarity'].apply(pd.Series)], axis=1).fillna('')
PSPECLANG = pd.concat([ PSPECLANG.drop(['VISITDATE'],axis=1), PSPECLANG['polarity'].apply(pd.Series)], axis=1)
PNONCONOTH = pd.concat([PNONCONOTH.drop(['VISITDATE'],axis=1), PNONCONOTH['polarity'].apply(pd.Series)], axis=1)
PSTRATOTH = pd.concat([ PSTRATOTH.drop(['VISITDATE'],axis=1), PSTRATOTH['polarity'].apply(pd.Series)], axis=1)
RSPNTOTH = pd.concat([ RSPNTOTH.drop(['VISITDATE'],axis=1), RSPNTOTH['polarity'].apply(pd.Series)], axis=1)
PRSPNDOTH = pd.concat([ PRSPNDOTH.drop(['VISITDATE'],axis=1), PRSPNDOTH['polarity'].apply(pd.Series)], axis=1)
SPECNOATTEMPT = pd.concat([SPECNOATTEMPT.drop(['VISITDATE'],axis=1),  SPECNOATTEMPT['polarity'].apply(pd.Series)], axis=1)
NCTPEROT = pd.concat([NCTPEROT.drop(['VISITDATE'],axis=1), NCTPEROT['polarity'].apply(pd.Series)], axis=1)
NCTTELOT = pd.concat([NCTTELOT.drop(['VISITDATE'],axis=1), NCTTELOT['polarity'].apply(pd.Series)], axis=1)
STRATOTH = pd.concat([STRATOTH.drop(['VISITDATE'],axis=1), STRATOTH['polarity'].apply(pd.Series)], axis=1)
#SPECLANG = pd.concat([SPECLANG.drop(['VISITDATE'],axis=1), SPECLANG['polarity'].apply(pd.Series)], axis=1)


CTOTHER['sentiment'] = CTOTHER['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
PSPECLANG['sentiment'] = PSPECLANG['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
PNONCONOTH['sentiment'] = PNONCONOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
PSTRATOTH['sentiment'] = PSTRATOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
RSPNTOTH['sentiment'] = RSPNTOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
PRSPNDOTH['sentiment'] = PRSPNDOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
SPECNOATTEMPT['sentiment'] = SPECNOATTEMPT['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
NCTPEROT['sentiment'] = NCTPEROT['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
NCTTELOT['sentiment'] = NCTTELOT['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
STRATOTH['sentiment'] = STRATOTH['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
#SPECLANG['sentiment'] = SPECLANG['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')


"""write CSV output """
CTOTHER.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\CTOTHER_sentiment.csv")
PSPECLANG.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\PSPECLANG_sentiment.csv")
PNONCONOTH.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\PNONCONOTH_sentiment.csv")
PSTRATOTH.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\PSTRATOTH_sentiment.csv")
RSPNTOTH.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\RSPNTOTH_sentiment.csv")
PRSPNDOTH.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\PRSPNDOTH_sentiment.csv")
SPECNOATTEMPT.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\SPECNOATTEMPT_sentiment.csv")
NCTPEROT.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\NCTPEROT_sentiment.csv")
NCTTELOT.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\NCTTELOT_sentiment.csv")
STRATOTH.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\STRATOTH_sentiment.csv")
#SPECLANG.to_csv(r"\\cdc.gov\csp_Project\CIPSEA_PII_NHIS_EXCHANGE\Census\PQN7nlp\corpus\sentiment_analysis\SPECLANG_sentiment.csv")



