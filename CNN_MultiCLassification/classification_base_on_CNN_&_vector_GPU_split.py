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



for i in [0,1,2,3,4,5,6,7,8,9]:
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.allDataFeatureList[i]["fileName"],
                                row_range = dfl.allDataFeatureList[i]["range"][0:200],
                                col = 0
                                ))
    
    label_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.allDataFeatureList[i]["fileName"],
                                row_range =dfl.allDataFeatureList[i]["range"][0:200],
                                col = 1
                                ))

# 字符串转为数字
label_list = [int(label) for label in label_list]

n_class = 10
label_list = fet.label_reflect2class(label_list,n_class)
# 将标签映射到类数以内

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

#扩充数据集
#word_set_list, label_list = fet.expand_content(word_set_list,label_list,specifiedWordNum=200)

#去除600词以下并归一化为600词
#2023-3-19 for in range 有坑，对i的改动是不会影响下一次循环的
word_set_list, label_list = fet.new_normalization_word_number(word_set_list,label_list,specifiedWordNum=200,thresholdWordNum=20)
print("归一化完成")
for word_set in word_set_list:
    print(len(word_set))

# 将数字列表转为ndarray
label_list:np.ndarray = fet.list_2_ndarray(label_list)

import time
start_word2vec = time.time()
# 将单词列表转化为词向量
word_gather_vec = fet.wordSet_to_Matrix(nlp,word_set_list,is_flat=False)
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
#2023-3-24
#这是因为tensorflow仍然采用了CPU运算
x_train = cupy.asnumpy(x_train)
x_test = cupy.asnumpy(x_test)

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,BatchNormalization
from keras.regularizers import l2

# 定义数据的输入形状
#input_shape = (200, 300)
input_shape = x_train.shape[1:]

# 定义数据分类的类的数量
num_classes = 10

# 定义CNN模型
model = Sequential()

model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape,kernel_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
# 添加一个Flatten层，将前一层的输出转换为一维矢量
model.add(Flatten())

# 添加一个具有64个单元和relu激活函数的全连接密集层
model.add(Dense(units=64, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(units=64, activation='relu'))
# 添加一个具有softmax激活函数的输出层，将我们的数据分类为num_classes
model.add(Dense(units=num_classes, activation='softmax'))

# 用分类交叉熵损失函数和adam优化器来编译模型categorical_crossentropy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

#保存模型
#model.save("./model/model_10_200_200_0416")

import tensorflow as tf
tf.keras.models.save_model(model,"./model/model_10_200_200_tf_0417.h5")
# 对测试集预测得到预测结果
y_pred = model.predict(x_test)

from keras.metrics import accuracy
# 计算损失和准确率
scores = model.evaluate(x_test,y_test)

#打印损失和准确率
#print(scores)
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
###################################
#          保存pred                #
##################################
file_path = './final/label_and_pred_04162359.txt'
with open(file_path,mode='w',encoding='utf-8') as file_obj:
    file_obj.write("label\tpred\n")
    idx:int = 0
    while(idx < len(y_test)):
        file_obj.write("%d\t" %(y_test[idx]))
        for pred in y_pred[idx]:
            file_obj.write("%f\t" %(pred))
        file_obj.write("\n")
        idx+=1
    
###################################
#          AUX绘图                #
##################################

# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# from sklearn.metrics import roc_curve, auc
# from scipy import interp
# from sklearn.preprocessing import LabelBinarizer

# n_classes = 10

# #标签改为独热编码
# label_binarizer = LabelBinarizer().fit(y_test)
# y_onehot_test = label_binarizer.transform(y_test)


# # 计算每一类的ROC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_pred[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
 
# # micro（方法二）
# fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_pred.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
 
# # macro（方法一）
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# # Finally average it and compute AUC
# mean_tpr /= n_classes
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
 
# # Plot all ROC curves
# lw=2
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
 
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
 
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
 
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('multi-calss ROC')
# plt.legend(loc="lower right")
# #plt.show()
# plt.savefig('./final/figs/auc.png')

# # 计算pr

# from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve,average_precision_score
# import matplotlib.pyplot as plt
# from itertools import cycle

# # For each class
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(y_onehot_test[:, i], y_pred[:, i])
#     average_precision[i] = average_precision_score(y_onehot_test[:, i], y_pred[:, i])

# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(
#     y_onehot_test.ravel(), y_pred.ravel()
# )
# average_precision["micro"] = average_precision_score(y_onehot_test, y_pred, average="micro")

# # setup plot details
# colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

# _, ax = plt.subplots(figsize=(7, 8))

# f_scores = np.linspace(0.2, 0.8, num=4)
# lines, labels = [], []
# for f_score in f_scores:
#     x = np.linspace(0.01, 1)
#     y = f_score * x / (2 * x - f_score)
#     (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
#     plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

# display = PrecisionRecallDisplay(
#     recall=recall["micro"],
#     precision=precision["micro"],
#     average_precision=average_precision["micro"],
# )
# display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

# for i, color in zip(range(n_classes), colors):
#     display = PrecisionRecallDisplay(
#         recall=recall[i],
#         precision=precision[i],
#         average_precision=average_precision[i],
#     )
#     display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

# # add the legend for the iso-f1 curves
# handles, labels = display.ax_.get_legend_handles_labels()
# handles.extend([l])
# labels.extend(["iso-f1 curves"])
# # set the legend and the axes
# ax.set_xlim([0.0, 1.0])
# ax.set_ylim([0.0, 1.05])
# ax.legend(handles=handles, labels=labels, loc="best")
# ax.set_title("Extension of Precision-Recall curve to multi-class")

# #plt.show()
# plt.savefig('./final/figs/pr.png')







