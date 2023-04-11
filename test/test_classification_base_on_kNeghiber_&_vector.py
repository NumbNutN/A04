####################################################################################
#                         基于词袋模型和线性分类器的简单分类                          #
#  -- Get start；                                                                  #
#       filename : csv数据集文件                                                    #
#       data_range : 加载数据集的范围                                                #
#       label_str_lst : 依据标签编号从小到大排序的字符标签数组，                       #
#                        编号定义详见classification_tool Class Label                 #
#  -- Info: 默认训练测试集82分                                                       #
#  -- Author: Created by LGD on 2023-3-9                                           #                                                              #
####################################################################################

import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import spacy

####################################################################################
#                                     数据选择                                      #
####################################################################################


dfl = fet.DataFeature()

text_list = []
label_list = []
# 3分类
for i in range(0,3):
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:180],
                                col = 1))
    label_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:180],
                                col = 2))
    
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
cnt:int = 0
ori_len:int = len(word_set_list)
for i in range(ori_len):
    if len(word_set_list[i]) < 600:
        word_set_list.pop(i)
        label_list.pop(i)
        cnt += 1
    if i == ori_len -cnt - 1:
        break
    

word_set_list = [word_set[0:600] for word_set in word_set_list]

# 将单词列表转化为词向量
word_gather_vec = fet.wordSet_to_Matrix(nlp,word_set_list,is_flat=True)

print("词向量转换完成")

print(
    "词向量矩阵类型:"+str(type(word_gather_vec[0])),
    "列表类型"+str(type(word_gather_vec))
    )
for i in range(len(word_gather_vec)):
    print(
        "矩阵维度:",
        i,
        word_gather_vec[i].shape
        )



# low_dim_embedded = []
# for embedded in word_gather_vec:
#     low_dim_embedded.append(fet.lower_dimension(embedded))


# 切分训练集和测试集
# 0.8 & 0.2
# 数据测试集切分
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = \
train_test_split(word_gather_vec,label_list,
                 test_size=0.2,random_state=0)
print("切分完成", flush=True)

# knn 分类模型

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(x_train,y_train)

# 对测试集预测得到预测结果
y_pred = clf.predict(x_test)

# 混淆矩阵
from sklearn.metrics import confusion_matrix
# 计算预测结果和真实结果的混淆矩阵
cnf_matrix = confusion_matrix(y_test,y_pred)


label_str_lst = ["信贷","刷单","婚恋"]
from tool import evaluation_tool as elt
# 通过混淆矩阵计算准确率
accuracy_lst = elt.evaluate_accuracy(cnf_matrix)
callback_lst = elt.evaluate_callback(cnf_matrix)
elt.print_format_accuracy_and_callback(label_str_lst,cnf_matrix)










