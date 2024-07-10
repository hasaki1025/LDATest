# Gensim
import heapq
import os.path

import gensim
from gensim.test.utils import datapath
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

from collections import Counter
from numpy import exp


def get_doc_by_id(data_lemmatized, doc_id):
    return ' '.join(data_lemmatized[doc_id])


def get_topic_doc(doc_list, data_lemmatized):
    docs = [get_doc_by_id(data_lemmatized, i.doc_id) for i in doc_list]
    return ' '.join(docs)


def relevance_to_color(relevance):
    """
    根据相关度返回HTML颜色代码（从绿色到红色渐变）

    参数:
    - relevance: 浮点数，表示相关度，范围在0到1之间

    返回:
    - 字符串，表示HTML颜色代码
    """
    # 映射相关度到颜色渐变
    # 当relevance为0时，使用绿色（#00FF00），当为1时，使用红色（#FF0000）
    # 这里我们使用线性插值来得到中间的颜色
    red = int(255 * relevance)
    green = 255 - red
    blue = 0  # 保持蓝色为0，以实现绿到红的渐变

    return f'color: rgb({red}, {green}, {blue});'


def get_topic_map(lda_model, corpus, list_size):
    doc_topics = lda_model.get_document_topics(corpus, minimum_probability=0)
    map = {}

    class wrapper:
        def __init__(self, doc_id, score):
            self.doc_id = doc_id
            self.score = score

        def __lt__(self, other):
            return self.score < other.score

        def __str__(self):
            return '{' + str(self.doc_id) + ':' + str(self.score) + '}'

    for topic_id in range(lda_model.num_topics):
        doc_list = []
        for doc_id, doc_dist in enumerate(doc_topics):
            score = doc_dist[topic_id]
            heapq.heappush(doc_list, wrapper(doc_id, score))
            if len(doc_list) >= list_size:
                heapq.heappop(doc_list)
        map[topic_id] = doc_list

    return map


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


# TODO 是否存在三元词组
def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# TODO 词形还原后是否删去某些词
def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def create_dictionary(texts):
    return corpora.Dictionary(texts)


def get_word_freq(texts, words=10):
    flat_texts = [word for doc in texts for word in doc]
    return Counter(flat_texts).most_common(words)


def text2bow(texts, dictionary):
    return [dictionary.doc2bow(text) for text in texts]


# 返回值：文本向量、词典、词形还原后的语料
def pre_data(data, language):
    data_words = sent_to_words(data)
    # TODO 阈值设置
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # TODO 和bigram的区别
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


def LDA(corpus, id2word, data_lemmatized, num_topics, save_model=True, model_file='model'):
    model_file = datapath(model_file)
    if os.path.exists(model_file):
        lda_model = gensim.models.ldamodel.LdaModel.load(model_file)
    else:
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
    return lda_model
