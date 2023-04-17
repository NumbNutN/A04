import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import numpy as np
import csv
import spacy
import copy

dfl = fet.DataFeature()

spacy.prefer_gpu()

url_list = []
text_list = []

csv.field_size_limit(20000*6000)


url_list.extend(fet.read_csv_context(
                                filename="./final/data/lost_test_with_content_0417.csv",
                                row_range = range(0,727),
                                col = 0))

text_list.extend(fet.read_csv_context(
                                filename="./final/data/lost_test_with_content_0417.csv",
                                row_range = range(0,727),
                                col = 1))

# text_list = copy.deepcopy(ori_text_list)
# url_list = copy.deepcopy(ori_url_list)


# 加载分词工具
nlp = spacy.load('zh_core_web_md')


batch_list = [range(0,500),
              range(500,1000),
              range(1000,1500),
              range(1500,2000),
              range(2000,2500),
              range(2500,3000),
              range(3000,3500),
              range(3500,4000),
              range(4000,4552)]

for batch_range in batch_list:

    split_text_list = []
    for idx in batch_range:
        if idx < len(text_list):
            split_text_list.append(text_list[idx])

    split_url_list = []
    for idx in batch_range:
        if idx < len(url_list):
            split_url_list.append(url_list[idx])

    ################
    #new
    record_list = []
    idx:int = 0
    for i in range(len(split_text_list)):
        if split_text_list[idx] == "Connect Failed" or split_text_list[idx] == "Nothing":
            record_list.append([i,split_url_list[idx],split_text_list[idx]])
            split_url_list.pop(idx)
            split_text_list.pop(idx)
        else:
            idx+=1

    split_word_set_list = fet.split_to_word_set_from_sentence(nlp,split_text_list)
    print("分词完成")

    #去除停用词
    from spacy.lang.zh.stop_words import STOP_WORDS

    split_word_set_list = fet.word_set_throw_stop_word(split_word_set_list,list(STOP_WORDS))
    print("去除停用词完成")
        


    #去除600词以下并归一化为600词
    #2023-3-19 for in range 有坑，对i的改动是不会影响下一次循环的
    new_split_word_set_list, split_url_list = fet.new_normalization_word_number(split_word_set_list,labelList = split_url_list,specifiedWordNum=200,thresholdWordNum=0)
    print("归一化完成")

    # 将单词列表转化为词向量
    word_gather_vec = fet.wordSet_to_Matrix(nlp,new_split_word_set_list,is_flat=False)
    print("词向量转换完成")

    import cupy
    #转换为CPU类型
    #2023-3-24
    #这是因为tensorflow仍然采用了CPU运算
    word_gather_vec = cupy.asnumpy(word_gather_vec)


    from keras.models import Sequential
    import keras
    import tensorflow as tf
    #model:Sequential = tf.keras.models.load_model("./model/model_10_200_200_tf_0416")
    #model = tf.saved_model.load("./model/model_10_10_200_tf_0417")
    #model = keras.models.load_model("./model/model_10_200_200_tf_0416")
    model = keras.models.load_model("./model/model_10_200_200_tf_0417.h5")
    # 对测试集预测得到预测结果
    y_pred = model.predict(word_gather_vec)

    #获得预测标签
    label_list = []
    for pred in y_pred:
        label_list.append(np.argmax(pred))

    #标签重映射

    label_list = [label+2 for label in label_list]

    for record in record_list:
        split_url_list:list.insert(record[0],record[1])
        label_list:list.insert(record[0],record[2])


    import pandas as pd
    data = {
        'url':split_url_list,
        'label':[label for label in label_list]
    }

    df = pd.DataFrame(data)
    #df = df.sort_values(by='label',ascending=True)
    df.to_csv("./final/data/lost_test_pred_0417.csv",sep=',',mode='a',header=False,index=False,encoding='utf-8')