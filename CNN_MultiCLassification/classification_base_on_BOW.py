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

import spacy
from sklearn.feature_extraction.text import CountVectorizer

# 数据测试集切分
from sklearn.model_selection import train_test_split

# 逻辑回归
from sklearn.linear_model import LogisticRegression

# 混淆矩阵
from sklearn.metrics import confusion_matrix

# 特征处理工具库
from tool import feature_extraction_tool as fet

# 分类问题工具库
from tool import classification_tool as cft

# 评估工具库
from tool import evaluation_tool as elt


####################################################################################
#                                     用户定义                                      #
####################################################################################

# 加载网页文本快照数据 train.csv
filename = "./data/train2.csv"
# 数据集范围 定义从csv文件中提取的数据的范围
data_range = range(650,800)
# 数据的第651 - 800 个词条

label_str_lst = ["购物消费","婚恋交友","信贷理财","刷单诈骗"]

# 训练集信息类
dfl = fet.DataFeature()
# 3分类

#数据集
text_list = []
#数据标签
label_list = []

# update: 2023-3-18 增加了数据集数量并增加
for i in range(0,3):
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:20],
                                col = 1))
    label_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:20],
                                col = 2))
    
for i in range(3,4):
    text_list.extend(fet.read_xlsx_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:20],
                                col = 1))
    label_list.extend(fet.read_xlsx_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:20],
                                col = 2))

# # 从csv表格中读取第2-4行的文本数据
# text_list = fet.read_csv_context(filename,data_range)
# # ["dogcatfish row2 col2","dogcatcat row3 col2","fishbird row4 col2"]

# # 获取数据标签
# label_lst = cft.get_label_from_csv(filename,data_range)
print("标签获取完成", flush=True)

# [1,2,2,0,0,3,4 ... ]


####################################################################################
#                                     特征提取                                      #
####################################################################################


# 加载分词工具
nlp = spacy.load('zh_core_web_md')

# 对文本进行分词
word_list = fet.split_word_from_sentence_array(nlp,text_list)
# ["dog cat fish row2 col2","dog cat cat row3 col2","fish bird row4 col2"]
print("分词完成", flush=True)


# 加载词袋模型分析器  定义最高频的5000个词语作为词袋的维度  /  一共12000多词
cv = CountVectorizer(max_features=5000)
# 接收词袋模型
cv_fit = cv.fit_transform(word_list)
print("词袋生成完成", flush=True)

# print(cv_fit.toarray())

####################################################################################
#                                      分类                                        #
####################################################################################



# 切分训练集和测试集
# 0.8 & 0.2
x_train, x_test,y_train, y_test = \
train_test_split(cv_fit.toarray(),label_list,
                 test_size=0.2,random_state=0)
print("切分完成", flush=True)

# 逻辑回归
LR_model = LogisticRegression()
# 传入训练集和标签进行训练
print("开始训练", flush=True)
LR_model = LR_model.fit(x_train,y_train)
print("训练结束", flush=True)

####################################################################################
#                                      评估                                        #
####################################################################################

# 对测试集预测得到预测结果
y_pred = LR_model.predict(x_test)
# 计算预测结果和真实结果的混淆矩阵
cnf_matrix = confusion_matrix(y_test,y_pred)
########################
#             预  测    #
#           正      负  #
# 真   正   TP     FN   #
##  ##  ##  ##  ##  ##  #
# 实   负   FP     TN   #
#########################


# 通过混淆矩阵计算准确率
accuracy_lst = elt.evaluate_accuracy(cnf_matrix)
callback_lst = elt.evaluate_callback(cnf_matrix)
elt.print_format_accuracy_and_callback(label_str_lst,cnf_matrix)

#标签        准确率       召回率
#信贷理财      0.882353  0.647059
#婚恋交友      0.846154  0.846154

