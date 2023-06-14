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

label_list.extend(fet.read_csv_context(
                                filename="./data/all_content_strenghen_5_18.csv",
                                row_range = range(300000),
                                col = 1))

content_list.extend(fet.read_csv_context(
                                filename="./data/all_content_strenghen_5_18.csv",
                                row_range = range(300000),
                                col = 0))


# 加载分词工具
nlp = spacy.load('zh_core_web_md')

#关键词剔除
substr = ['没有找到','建设中']
content_list,label_list = fet.throw_context_including_spe_substr(content_list,label_list,substr)


import pandas as pd

data = {
    'text':content_list,
    'label':[int(label) for label in label_list]
}

df = pd.DataFrame(data)

class_hunlian_contentList = []
class_hunlian_labelList = []
cnt = 0

for row_index,row in df.iterrows():
    if(row['label'] == 2):
        class_hunlian_contentList.append(row['text'])
        class_hunlian_labelList.append(row['label'])
        df = df.drop(index=df.index[row_index-cnt])
        cnt+=1

data_hunlian = {
    'text':class_hunlian_contentList,
    'label':[int(label) for label in class_hunlian_labelList]
}
class2df = pd.DataFrame(data_hunlian)

subclass2df = class2df.sample(n=3000)

df = pd.concat([df,subclass2df])

df = df.sort_values(by='label',ascending=True)


content_list = df['text'].tolist()
label_list = df['label'].tolist()

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = \
train_test_split(content_list,label_list,
                 test_size=0.2,random_state=0)

sentence_list = []
new_label_list = []

split_sentence = ""

for i in range(len(content_list)):
    nlpobj = nlp(content_list[i])
    for sentence in nlpobj.sents:
    #     if (len(split_sentence) + len(sentence.text)) <= 126:
    #         split_sentence += ('[SEP]' + sentence.text)
    #     else:
    #         sentence_list.append(split_sentence)
    #         new_label_list.append(label_list[i])
    #         split_sentence = ""
    # if(split_sentence != ""):
    #     sentence_list.append(split_sentence)
    #     split_sentence = ""
    #     new_label_list.append(label_list[i])

        sentence_list.append(sentence.text)
        new_label_list.append(label_list[i])


import pandas as pd
data = {
    'text':sentence_list,
    'label':[int(label) for label in new_label_list]
}

df = pd.DataFrame(data)
df = df.sort_values(by='label',ascending=True)
df.to_csv("./data/all_content_strenghen_5_18_2_sentence_less_hunlian_train.csv",sep=',',mode='w',header=False,index=False,encoding='utf-8')



label_list = new_label_list

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