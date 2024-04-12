#%%
#tokenized_sents = pd.DataFrame(texts_out)
#tokenized_sents.to_parquet(r"C:\_harvester\data\lda-models\2010s_html.json\tokenized_sents-w-bigrams.parquet")

import pickle
with open(r"C:\_harvester\data\lda-models\2010s_html.json\word2vec-2010s\tokenized-texts-out\tokenized_sents-w-bigrams.pkl", "rb") as fp:
    texts_out = pickle.load(fp)
print(texts_out[:25])


#%%
import torch
from tqdm import tqdm
import csv
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

### choose the callbacks classes to import
import logging
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric
from gensim.models import CoherenceModel
import numpy as np

#%%

def lda_topic_analysis(documents):
    # Set up GPU device
    device = torch.device("cuda")

    # Read documents into method with progress bar
    with tqdm(total=len(documents), desc="Reading Documents") as pbar:
        dictionary = Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        pbar.update(len(documents))

    # Set up the callbacks loggers
    perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
    convergence_logger = ConvergenceMetric(logger='shell')
    coherence_cv_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence = 'c_v', texts = texts_out)
    
    # Define range of topics, alpha, and beta values to test
    num_topics_range = range(100, 505, 5)

    # Alpha parameter
    alpha_range = list(np.arange(0.01, 1, 0.3))
    alpha_range.append('symmetric')
    alpha_range.append('asymmetric')

    # Beta parameter
    beta_range = list(np.arange(0.01, 1, 0.3))
    beta_range.append('symmetric')
    beta_range.append('asymmetric')

    # Initialize CSV file to store results
    with open(r'C:\_harvester\data\lda-models\2010s_html.json\results\topic_analysis_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Num Topics', 'Alpha', 'Beta', 'Coherence', 'Log Perplexity', 'Convergence Score'])

        # Generate models with status bar
        with tqdm(total=len(num_topics_range) * len(alpha_range) * len(beta_range), desc="Generating Models") as pbar:
            for num_topics in num_topics_range:
                for alpha in alpha_range:
                    for beta in beta_range:
                        
                        # Add text to logger to indicate new model
                        logging.debug(f'Start of model: {num_topics} topics')

                        lda_model = LdaModel(corpus=corpus,
                                             id2word=dictionary,
                                             num_topics=num_topics,
                                             alpha=alpha ,
                                             eta=beta ,
                                             passes=10,
                                             iterations=100,
                                             random_state=42,
                                             per_word_topics=True,
                                             callbacks=[convergence_logger, perplexity_logger, coherence_cv_logger])
                        
                        lda_model.to(device)  # Move model to GPU

                        # Add text to logger to indicate end of this model
                        logging.debug(f'End of model: {num_topics} topics\n')


                        coherence_model = CoherenceModel(model=lda_model,
                                                         texts=documents,
                                                         dictionary=dictionary,
                                                         coherence='c_v')
                        coherence_score = coherence_model.get_coherence()
                        log_perplexity_score = lda_model.log_perplexity(corpus)
                        convergence_score = lda_model.bound(corpus)

                        # Save the model
                        model_name = f"C:/_harvester/data/lda-models/2010s_html.json/lda-corpusset(0)/lda_model_{num_topics}_{alpha}_{beta}.model"
                        lda_model.save(model_name)

                        # Write results to CSV file
                        writer.writerow([num_topics, alpha, beta, coherence_score, log_perplexity_score, convergence_score])

                        pbar.update(1)
    logging.shutdown()
    return lda_model


#%%
# If the file already exists, the log will continue rather than being overwritten.
# create log file to view log perplexity, topics, etc.
import logging
logging.basicConfig(filename=f"C:/_harvester/data/lda-models/2010s_html.json/model_callbacks.log",
            format="%(asctime)s:%(levelname)s:%(message)s",
            level=logging.NOTSET)

lda = lda_topic_analysis(texts_out)


# %%
import logging
logging.shutdown()
# %%
