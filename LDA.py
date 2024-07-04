# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import re
from langdetect import detect, DetectorFactory

import pandas as pd
from pprint import pprint
# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
from nltk.corpus import stopwords
import warnings

from numpy import exp


# pyLDAvis.enable_notebook()


def load_data(file_path, keep_url=False):
    data = pd.read_excel(file_path, engine='openpyxl').iloc[:, 10].dropna(how='any')
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    # 如果无需保留URL则将其替换为空字符串
    if not keep_url:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        data = [re.sub(url_pattern, "", sent) for sent in data]

    # 删除所有NIX行
    data = [sent for sent in data if sent != "NIX"]
    return data


def sent_to_words(sentences):
    return [gensim.utils.simple_preprocess(str(sentence), deacc=True) for sentence in sentences]


def remove_stopwords(texts, stop_words):
    out = []
    for text in texts:
        out.append([word for word in simple_preprocess(str(text)) if word not in stop_words])
    return out


def make_bigrams(data_words, bigram_mod):
    return [bigram_mod[doc] for doc in data_words]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def create_dictionary(texts):
    return corpora.Dictionary(texts)


def text2bow(texts, dictionary):
    return [dictionary.doc2bow(text) for text in texts]


# 返回值：文本向量、词典、词形还原后的语料
def pre_data(data, language):
    data_words = sent_to_words(data)
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    if language == 'en':
        stop_words = stopwords.words('english')
        nlp = spacy.load("en_core_web_sm", disable=['parser'])
    else:
        stop_words = stopwords.words('german')
        nlp = spacy.load('de_core_news_sm', disable=['parser'])

    data_words_nostops = remove_stopwords(data_words, stop_words)
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    return lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # id2word = corpora.Dictionary(data_lemmatized)
    # texts = data_lemmatized
    # corpus = [id2word.doc2bow(text) for text in texts]


def LDA(corpus, id2word, data_lemmatized, num_topics):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # pprint(lda_model.print_topics())

    perplexity = pow(2, -lda_model.log_perplexity(corpus))
    print('\nPerplexity: ', perplexity)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    return vis, perplexity, coherence_lda
