##########################################
#       加载模型                         #
##########################################

import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import spacy
import numpy as np

for path in sys.path:
    print(path)


####################################################################################
#                                     数据选择                                      #
####################################################################################


dfl = fet.DataFeature()
spacy.prefer_gpu()
text_list = []
label_list = []
# 3分类
# 字符串标签转数字
from tool import classification_tool as ct

# for i in range(0,3):
#     text_list.extend(fet.read_csv_context(
#                                 filename="./data/"+dfl.dataFeatureList[i]["fileName"],
#                                 row_range = dfl.dataFeatureList[i]["range"][170:200],
#                                 col = 1))
    
#     # 由于kears要求使用数字作为标签
#     label_list.extend(ct.get_label_from_csv(
#                                 filename="./data/"+dfl.dataFeatureList[i]["fileName"],
#                                 row_range =dfl.dataFeatureList[i]["range"][170:200]
#                                 ))
for i in [0,1,2]:
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:30],
                                col = 1))
    
    # 由于kears要求使用数字作为标签
    label_list.extend(ct.get_label_from_csv(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:30]
                                ))
for i in [10,12,15]:
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:30],
                                col = 1
                                ))
    
    label_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:30],
                                col = 2
                                ))

# 字符串转为数字
label_list = [int(label) for label in label_list]

# 加载分词工具
nlp = spacy.load('zh_core_web_md')

'''
# 对文本进行分词
split_text_list = fet.split_word_arr(nlp,text_list)
# ["dog cat fish row2 col2","dog cat cat row3 col2","fish bird row4 col2"]
print("分词完成")

# 将spacy产生的分词文本变为单词列表
# TODO:未进行常用词过滤和重复词的消去
# TODO:可以在spacy实现单词列表的原始实现，没必要由这种形式转回来
word_set_list = []
for i in range(len(split_text_list)):
    word_set_list.append(fet.split_word_sentence_to_split_word_list(split_text_list[i]))
#[["dog", "cat", "fish", "row2"], ["col2","dog", "cat", "cat", "row3", "col2"],["fish", "bird", "row4", "col2"]]
'''

word_set_list = fet.split_to_word_set_from_sentence(nlp,text_list)
print("分词完成")

#去除停用词
from spacy.lang.zh.stop_words import STOP_WORDS

fet.word_set_throw_stop_word(word_set_list,list(STOP_WORDS))
print("去除停用词完成")


#去除600词以下并归一化为600词
#2023-3-19 for in range 有坑，对i的改动是不会影响下一次循环的
word_set_list, label_list = fet.normalization_word_number(word_set_list,label_list,600)
print("归一化完成")


# 将数字列表转为ndarray
label_list:np.ndarray = fet.list_2_ndarray(label_list)

import time
start_word2vec = time.time()
# 将单词列表转化为词向量
word_gather_vec = fet.wordSet_to_Matrix(nlp,word_set_list,is_flat=False)
end_word2vec = time.time()
print("词向量转换完成")
print("词向量转换用时%d" %(end_word2vec-start_word2vec))

# 切分训练集和测试集
# 0.8 & 0.2
# 数据测试集切分

import cupy

#转换为CPU类型
#2023-3-24
#这是因为tensorflow仍然采用了CPU运算
word_gather_vec = cupy.asnumpy(word_gather_vec)

from keras.models import Sequential

import keras

model:Sequential = keras.models.load_model("./model/model_6_200_0405")
score = model.evaluate(word_gather_vec,label_list)
print(score)
#预测模型

