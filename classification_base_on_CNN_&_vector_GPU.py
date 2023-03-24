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


from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import spacy
import numpy as np

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

for i in range(0,3):
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:40],
                                col = 1))
    
    # 由于kears要求使用数字作为标签
    label_list.extend(ct.get_label_from_csv(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:40]
                                ))


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

word_set_list = fet.split_word_single_arr(nlp,text_list)
print("分词完成")

#去除停用词
from spacy.lang.zh.stop_words import STOP_WORDS

fet.text_list_throw_stop_word(word_set_list,list(STOP_WORDS))
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
word_gather_vec = fet.wordGatherList_to_Matrix(nlp,word_set_list,is_flat=False)
end_word2vec = time.time()
print("词向量转换完成")
print("词向量转换用时%d" %(end_word2vec-start_word2vec))



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

# 切分训练集和测试集
# 0.8 & 0.2
# 数据测试集切分


from sklearn.model_selection import train_test_split

import cupy
x_train, x_test,y_train, y_test = \
train_test_split(word_gather_vec,label_list,
                 test_size=0.2,random_state=0)
print("切分完成", flush=True)

#转换为CPU类型
x_train = cupy.asnumpy(x_train)
x_test = cupy.asnumpy(x_test)

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# 定义数据的输入形状
input_shape = (600, 300)

# 定义数据分类的类的数量
num_classes = 12

# 定义CNN模型
model = Sequential()

# 添加一个1D卷积层，有32个过滤器，核大小为3，并有relu激活函数
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))

# 添加一个MaxPooling1D层，池大小为2
model.add(MaxPooling1D(pool_size=2))

# 添加一个Flatten层，将前一层的输出转换为一维矢量
model.add(Flatten())

# 添加一个具有64个单元和relu激活函数的全连接密集层
model.add(Dense(units=64, activation='relu'))

# 添加一个具有softmax激活函数的输出层，将我们的数据分类为num_classes
model.add(Dense(units=num_classes, activation='softmax'))

# 用分类交叉熵损失函数和adam优化器来编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#训练
model.fit(x_train, y_train, epochs=11, validation_data=(x_test, y_test))


# 对测试集预测得到预测结果
y_pred = model.predict(x_test)

from keras.metrics import accuracy
# 计算损失和准确率
scores = model.evaluate(x_test,y_test)

#打印损失和准确率
print(scores)
# accu = accuracy(y_test,y_pred)

# # 混淆矩阵
# from sklearn.metrics import confusion_matrix
# # 计算预测结果和真实结果的混淆矩阵
# cnf_matrix = confusion_matrix(y_test,y_pred)


# label_str_lst = ["信贷","刷单","婚恋"]
# from tool import evaluation_tool as elt
# # 通过混淆矩阵计算准确率
# accuracy_lst = elt.evaluate_accuracy(cnf_matrix)
# callback_lst = elt.evaluate_callback(cnf_matrix)
# elt.print_format_accuracy_and_callback(label_str_lst,cnf_matrix)










