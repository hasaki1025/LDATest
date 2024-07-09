# 德语单独训练
import os.path
import html

import gensim.models
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt

import pyLDAvis

# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import LDA
import LanguageDetect as ld


def show_and_save_vis(vis, filename='output.html'):
    pyLDAvis.save_html(vis, filename)
    pyLDAvis.show(vis, local=False)


def write_out_as_html(text, filename):
    # 使用html.escape()函数转义特殊字符
    # escaped_string = html.escape(text)
    text_pre = """ <!DOCTYPE html>  
    <html lang="de">  
<head>  
    <meta charset="UTF-8">  
    <title>topic</title>  
</head>   """
    text_last = """ </html>"""
    # 将转义后的字符串写入HTML文件
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text_pre)
        file.write(text)
        file.write(text_last)


def train(num_topics, use_en=False, save_model=True, model_file='model'):
    file = 'data/LDAData.xlsx'
    data = LDA.load_data(file)
    datas = ld.language_detect(data)
    data_lemmatized = LDA.pre_data(datas['de'], 'de')
    if use_en:
        data_lemmatized = LDA.pre_data(datas['en'], 'en')
        model_file = model_file + '_hybrid'
    dictionary = LDA.create_dictionary(data_lemmatized)
    corpus = LDA.text2bow(data_lemmatized, dictionary)
    lda_model = LDA.LDA(corpus, dictionary, data_lemmatized, num_topics, save_model, model_file)

    perplexity = lda_model.log_perplexity(corpus)
    print('\nPerplexity: ', perplexity)
    if os.path.exists(model_file + '_CoherenceModel'):
        coherence_model_lda = gensim.models.CoherenceModel.load(model_file + '_CoherenceModel')
    else:
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary,
                                             coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    print('\nCoherence Score: ', coherence_lda)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

    if save_model:
        lda_model.save(model_file)

    print(LDA.get_topic_map(lda_model, corpus, 10))

    def text2HTML(text, special_words):
        html_text = ''
        words = text.split()
        i = 0
        while i < len(words):
            word = words[i]
            if word in special_words:
                # 获取颜色并添加HTML标签
                color = LDA.relevance_to_color(special_words[word])
                html_text += f'<span style="{color}">{word}</span>'
            else:
                html_text += word
            if i < len(words) - 1:
                html_text += ' '
            i += 1
        return html_text

    map = LDA.get_topic_map(lda_model, corpus, 10)
    for topic_id, topic_docs in map.items():
        topic_html_text = ''
        sp_words = {dictionary[word_id]: prob for word_id, prob in lda_model.get_topic_terms(topic_id)}
        print(sp_words)
        for doc_wrapper in topic_docs:
            text = data[doc_wrapper.doc_id]
            text = '<p>' + text2HTML(text, sp_words) + '</p>'
            topic_html_text = topic_html_text + text

        write_out_as_html(topic_html_text, f'topic{topic_id}Doc.html')
    return vis, perplexity, coherence_lda


if __name__ == '__main__':
    vis, plx, coherence = train(10, True)
    show_and_save_vis(vis, filename='output.html')
