import pyLDAvis

# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import LDA
import LanguageDetect as ld


def show_and_save_vis(vis, filename='output.html'):
    pyLDAvis.save_html(vis, filename)
    pyLDAvis.show(vis)


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    file = 'data/LDAData.xlsx'
    data = LDA.load_data(file)
    datas = ld.language_detect(data)
    data_lemmatized = LDA.pre_data(datas['en'], 'en') +  LDA.pre_data(datas['de'], 'de')
    dictionary = LDA.create_dictionary(data_lemmatized)
    corpus = LDA.text2bow(data_lemmatized, dictionary)
    vis = LDA.LDA(corpus, dictionary, data_lemmatized)
    print(2)
