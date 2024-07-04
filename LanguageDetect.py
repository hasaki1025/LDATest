import LDA
from langdetect import detect, DetectorFactory


def language_detect(texts):
    result = {}
    for sentence in texts:
        detected_language = detect(sentence)
        if detected_language not in result:
            result[detected_language] = []
        result[detected_language].append(sentence)
    return result



