import csv

import numpy as np
import LanguageDetect as ld

import LDA
from main import train


class edge:

    def __init__(self):
        self.source = 0
        self.target = 0

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __hash__(self):
        return hash((self.source, self.target))

    def __eq__(self, __value):
        return self.source == __value.source and self.target == __value.target


def generate_graph(datas, hot_words, dict):
    edge_map = {}
    for doc in datas:
        doc_hot_words = []
        for word in doc:
            if word in hot_words:
                doc_hot_words.append(word)
        doc_h_len = len(doc_hot_words)
        for i in range(doc_h_len):
            for j in range(i + 1, doc_h_len):
                source_id = dict.token2id[doc_hot_words[i]]
                target_id = dict.token2id[doc_hot_words[j]]
                e = edge(source_id, target_id)
                if e in edge_map:
                    edge_map[e] = edge_map[e] + 1
                else:
                    edge_map[e] = 1

    return edge_map


def graph2csv(edge_map, dictionary, word_counter, nameMappingFile='name.csv', graphFile='graph.csv'):
    with open(nameMappingFile, "w", encoding="utf-8") as csvf1:
        writer1 = csv.DictWriter(csvf1, fieldnames=['id', 'label', 'weight'])
        writer1.writeheader()
        for word, frequency in word_counter:
            line = {'id': dictionary.token2id[word], 'label': word, 'weight': frequency}
            writer1.writerow(line)

    with open(graphFile, "w", encoding="utf-8") as csvf2:
        writer2 = csv.DictWriter(csvf2, fieldnames=['source', 'target', 'weight'])
        writer2.writeheader()
        for edge, weight in edge_map.items():
            line = {'source': edge.source, 'target': edge.target, 'weight': weight}
            writer2.writerow(line)


if __name__ == '__main__':
    file = 'data/LDAData.xlsx'
    data = LDA.load_data(file)
    datas = ld.language_detect(data)
    data_lemmatized = LDA.pre_data(datas['de'], 'de')
    lda_model, corpus, dictionary = train(10)
    map = LDA.get_topic_map(lda_model, corpus, 10)

    for topic_id, wrapper_list in map.items():
        doc_id_list = [wrapper.doc_id for wrapper in wrapper_list]
        topic_data = [data_lemmatized[doc_id] for doc_id in doc_id_list]
        word_freq = LDA.get_word_freq(topic_data)
        hot_words = [p[0] for p in word_freq]
        edge_map = generate_graph(topic_data, hot_words, dictionary)
        graph2csv(edge_map, dictionary, word_freq, f'data/topic_{topic_id}_name.csv',
                  f'data/topic_{topic_id}_graph.csv')
