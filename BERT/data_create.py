import sys
sys.path.append(".")
from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# 数据测试集切分
from sklearn.model_selection import train_test_split

import time
####################################################################################
#                                     数据选择                                      #
####################################################################################


dfl = fet.DataFeature()
text_list = []
label_list = []
# 3分类
# 字符串标签转数字

path = "./BERT/"

start = time.time()

for i in range(0,10):
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.allDataFeatureList[i]["fileName"],
                                row_range = dfl.allDataFeatureList[i]["range"][0:400],
                                col = 0))
    

    label_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.allDataFeatureList[i]["fileName"],
                                row_range =dfl.allDataFeatureList[i]["range"][0:400],col = 1
                                ))

label_list = [int(label) for label in label_list]

print("标签获取完成", flush=True)

# 加载分词工具
nlp = spacy.load('zh_core_web_md')

# 对文本进行分词
word_list = fet.split_word_from_sentence_array(nlp,text_list)
print("分词完成", flush=True)

# 拆分训练集和测试集
x_train, x_test,y_train, y_test = \
    train_test_split(word_list,label_list,
                 test_size=0.2,random_state=0)

# 拆分训练集和验证集
x_train,x_dev,y_train,y_dev = \
    train_test_split(x_train,y_train,test_size=0.25,random_state=0)

with open(path+"train.txt","w",encoding='utf-8') as file:
    for i in range(len(x_train)):
        file.write(x_train[i]+'\t'+str(y_train[i])+'\n')

with open(path+"test.txt","w",encoding='utf-8') as file:
    for i in range(len(x_train)):
        file.write(x_train[i]+'\t'+str(y_train[i])+'\n')

with open(path+"dev.txt","w",encoding='utf-8') as file:
    for i in range(len(x_train)):
        file.write(x_train[i]+'\t'+str(y_train[i])+'\n')
        
