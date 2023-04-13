import spacy
from tool import feature_extraction_tool as fet
from spacy.lang.zh.stop_words import STOP_WORDS
from keras.models import Sequential
import keras
import numpy as np

nlp:spacy.Language

def content_based_model_init():
    global nlp
    # 加载分词工具
    nlp = spacy.load('zh_core_web_md')
    


def content_based_model_predict(content:str,wordNum:int) -> int:
    """返回预测的标签对应的标号(0-12)

    Args:
        content (str): 输入文本

    Returns:
        int: 预测的标签对应的标号
        None: 输入不合法（字符不多于200词）
    """
    global nlp
    #句子转换为单词集
    wordSet = fet.split_to_word_set_from_sentence(nlp,content)
    #剔除常用词
    wordSet = fet.word_set_throw_stop_word(wordSet,STOP_WORDS)
    #归一化
    wordSet = fet.new_normalization_word_number(wordSet,specifiedWordNum=wordNum)
    #字列表转换为矩阵
    wordSetVector = fet.wordSet_to_Matrix(nlp,wordSet)
    #加载模型
    model:Sequential = keras.models.load_model("./model/model_6_20_200_0411")
    #开始预测
    predictVec = model.predict(wordSetVector)
    #返回概率最高标签
    return np.argmax(predictVec)


string = "金融"
content_based_model_init()
predict  = content_based_model_predict(string,200)
print(predict)