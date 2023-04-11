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
    

def get_content_based_model_predict_vector(content:str) -> 'np.ndarray | None': 
    """基于内容的分类模型接口

    Args:
        content (str): 输入文本

    Returns:
        np.ndarray: 返回对所有分类的预测向量
        None: 输入不合法（字符不多于200词）

    """
    global nlp
    #句子转换为单词集
    wordSet = fet.split_to_word_set_from_sentence(nlp,content)
    #剔除常用词
    wordSet = fet.word_set_throw_stop_word(wordSet,STOP_WORDS)
    #归一化
    wordSet = fet.normalization_word_number(wordSet,number=200)
    #如果数量小于200，返回NONE
    if(len(wordSet) == 0):
        return None
    #字列表转换为矩阵
    wordSetVector = fet.wordSet_to_Matrix(nlp,wordSet)
    #加载模型
    model:Sequential = keras.models.load_model("./model/model_6_20_200_0411")
    #开始预测
    predict = model.predict(wordSetVector)
    #结果存储在predict
    return predict

def content_based_model_predict(content:str) -> int:
    """返回预测的标签对应的标号(0-12)

    Args:
        content (str): 输入文本

    Returns:
        int: 预测的标签对应的标号
        None: 输入不合法（字符不多于200词）
    """
    vec = get_content_based_model_predict_vector(content)
    if(vec is not None):
        return np.argmax(vec)
    else:
        return vec

string = "金融"
content_based_model_init()
predict  = content_based_model_predict(string)
pass