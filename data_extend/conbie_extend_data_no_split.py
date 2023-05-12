import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import spacy
import numpy as np
import csv

####################################################################################
#                                汇总增强数据集                                      #
####################################################################################


dfl = fet.DataFeature()
spacy.prefer_gpu()
text_list = []
label_list = []
content_list = []
# 3分类
# 字符串标签转数字
from tool import classification_tool as ct


#补充数据集
label_list.extend(fet.read_csv_context(
                                filename="./data/more_supplement2_seq_0413.csv",
                                row_range = range(30000),
                                col = 2))

content_list.extend(fet.read_csv_context(
                                filename="./data/more_supplement2_seq_0413.csv",
                                row_range = range(30000),
                                col = 1))

#train1
label_list.extend(fet.read_csv_context(
                                filename="./data/more_haveContent0405.csv",
                                row_range = range(30000),
                                col = 2))

content_list.extend(fet.read_csv_context(
                                filename="./data/more_haveContent0405.csv",
                                row_range = range(30000),
                                col = 1))


#train2
label_list.extend(ct.read_csv_label_a2i(
                            filename="./data/train2.csv",
                            row_range =range(2,30000)
                            ))

content_list.extend(fet.read_csv_context(
                            filename="./data/train2.csv",
                            row_range = range(2,30000),
                            col = 1))

#train3
train3_label_list = fet.read_xlsx_context(
                                filename="./data/train3.xlsx",
                                row_range = range(30000),
                                col = 2)

def train3_label_cvt(label:int)->int:
    if label == 2:
        return 10
    if label == 1:
        return 11

train3_label_list = [int(label) for label in train3_label_list]

train3_label_list = [train3_label_cvt(label) for label in train3_label_list]


content_list.extend(fet.read_xlsx_context(
                                filename="./data/train3.xlsx",
                                row_range = range(30000),
                                col = 1))

label_list.extend(train3_label_list)


# 加载分词工具
nlp = spacy.load('zh_core_web_md')

word_set_list = fet.split_to_word_set_from_sentence(nlp,content_list)
print("分词完成")

#去除停用词
from spacy.lang.zh.stop_words import STOP_WORDS

word_set_list = fet.word_set_throw_stop_word(word_set_list,list(STOP_WORDS))
print("去除停用词完成")

#扩充数据集
#word_set_list, label_list = fet.expand_content(word_set_list,label_list,specifiedWordNum=200)


#小于30词被剔除
#2023-3-19 for in range 有坑，对i的改动是不会影响下一次循环的
word_set_list, label_list = fet.throw_lower_than_threshold(word_set_list,label_list,thresholdWordNum=30)
print("剔除少量词完成")

content_list = []
for word_set in word_set_list:
    content_list.append(''.join(word_set))
print("单词集转句子完成")

import pandas as pd

data = {
    'text':content_list,
    'label':[int(label) for label in label_list]
}

df = pd.DataFrame(data)
df = df.sort_values(by='label',ascending=True)
df.to_csv("./data/all_content_without_split.csv",sep=',',mode='w',header=False,index=False,encoding='utf-8')


info_list = []

oldlabel:str = ''
oldcnt:int=0
idx:int=0
for label in label_list:
    if label != oldlabel:
        info_list.append({'label':oldlabel,'count':oldcnt,'start':idx-oldcnt,'end':idx})
        oldcnt=0
        oldlabel=label
    oldcnt+=1
    idx+=1
pass