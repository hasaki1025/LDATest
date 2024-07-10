import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud, ImageColorGenerator  # 词云
import LanguageDetect as ld
import LDA
from main import train


def generate_word_cloud(word_freq, image_name='word_cloud.png'):
    alice_mask = np.array(Image.open('image/alice_mask.png'))
    font_path = "font/msyh.ttf"
    stop_words = set()

    wc = WordCloud(background_color="white",  # 设置背景颜色
                   max_words=2000,  # 词云显示的最大词数
                   mask=alice_mask,  # 设置背景图片
                   stopwords=stop_words,  # 设置停用词
                   font_path=font_path,  # 兼容中文字体，不然中文会显示乱码
                   )
    if isinstance(word_freq, dict):
        wc.generate_from_frequencies(word_freq)
    else:
        wc.generate(word_freq)
    wc.to_file(image_name)
    plt.imshow(wc, interpolation='bilinear')
    # interpolation='bilinear' 表示插值方法为双线性插值
    plt.axis("off")  # 关掉图像的坐标
    plt.show()


if __name__ == '__main__':
    file = 'data/LDAData.xlsx'
    data = LDA.load_data(file)
    datas = ld.language_detect(data)
    data_lemmatized = LDA.pre_data(datas['de'], 'de')
    lda_model, corpus, dictionary = train(10)
    print(lda_model.get_topic_terms(0))
    map = LDA.get_topic_map(lda_model, corpus, 10)
    for topic_id, doc_list in map.items():
        generate_word_cloud(LDA.get_topic_doc(doc_list, data_lemmatized), image_name=f'image/topic/topic{topic_id}.png')
