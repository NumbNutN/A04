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

ori_url_list = []
ori_text_list = []

csv.field_size_limit(20000*6000)

ori_url_list.extend(fet.read_csv_context(
                                filename="./final/data/test_with_text_0416.csv",
                                row_range = range(20000),
                                col = 0))

ori_text_list.extend(fet.read_csv_context(
                                filename="./final/data/test_with_text_0416.csv",
                                row_range = range(20000),
                                col = 1))

text_list = copy.deepcopy(ori_text_list)
url_list = copy.deepcopy(ori_url_list)


idx:int = 0
for i in range(len(text_list)):
    if text_list[idx] == "Connect Failed" or text_list[idx] == "Nothing":
        url_list.pop(idx)
        text_list.pop(idx)
    else:
        idx+=1

# 加载分词工具
nlp = spacy.load('zh_core_web_md')

word_set_list = fet.split_to_word_set_from_sentence(nlp,text_list)
print("分词完成")

#去除停用词
from spacy.lang.zh.stop_words import STOP_WORDS

word_set_list = fet.word_set_throw_stop_word(word_set_list,list(STOP_WORDS))
print("去除停用词完成")


#去除600词以下并归一化为600词
#2023-3-19 for in range 有坑，对i的改动是不会影响下一次循环的
word_set_list, label_list = fet.new_normalization_word_number(word_set_list,specifiedWordNum=200,thresholdWordNum=0)
print("归一化完成")


# 将单词列表转化为词向量
word_gather_vec = fet.wordSet_to_Matrix(nlp,word_set_list,is_flat=False)
print("词向量转换完成")


import cupy

#转换为CPU类型
#2023-3-24
#这是因为tensorflow仍然采用了CPU运算
word_gather_vec = cupy.asnumpy(word_gather_vec)


from keras.models import Sequential

import keras
import tensorflow as tf
model:Sequential = tf.keras.models.load_model("./model/model_10_200_200_tf_0416")

# 对测试集预测得到预测结果
y_pred = model.predict(word_gather_vec)

#获得预测标签
label_list = []
for pred in y_pred:
    label_list.append(np.argmax(pred))

#标签重映射

label_list = [label+2 for label in label_list]


import pandas as pd
data = {
    'url':url_list,
    'label':[int(label) for label in label_list]
}

df = pd.DataFrame(data)
#df = df.sort_values(by='label',ascending=True)
df.to_csv("./final/data/test_with_text_pred_0416.csv",sep=',',mode='w',header=False,index=False,encoding='utf-8')