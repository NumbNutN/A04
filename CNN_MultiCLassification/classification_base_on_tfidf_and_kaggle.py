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

start = time.time()

# for i in range(0,10):
#     text_list.extend(fet.read_csv_context(
#                                 filename="./data/"+dfl.allDataFeatureList[i]["fileName"],
#                                 row_range = dfl.allDataFeatureList[i]["range"][0:400],
#                                 col = 0))
    

#     label_list.extend(fet.read_csv_context(
#                                 filename="./data/"+dfl.allDataFeatureList[i]["fileName"],
#                                 row_range =dfl.allDataFeatureList[i]["range"][0:400],col = 1
#                                 ))

ori_train_texts = []
ori_train_labels = []

class_list = ["婚恋交友", "假冒身份" ,"钓鱼网站", "冒充公检法" ,"平台诈骗" ,"招聘兼职" ,"杀猪盘" ,"博彩赌博" ,"信贷理财" ,"刷单诈骗" ]
for class_name in class_list:
    ori_train_texts.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_train.csv",
                                    row_range = dfl.train_split_dataFeature[class_name]["range"][0:400],
                                    col = 0))

    ori_train_labels.extend(fet.read_csv_context(
                                    filename="/A04/bert_data/all_content_split_train.csv",
                                    row_range = dfl.train_split_dataFeature[class_name]["range"][0:400],
                                    col = 1))

text_list = []
label_list = []
for idx in range(len(ori_train_labels)):
    if(ori_train_labels[idx] != '1' and ori_train_labels[idx] != '12'):
        text_list.append(ori_train_texts[idx])
        label_list.append(ori_train_labels[idx])
# train_labels = filter(remove_label_reward,train_labels)


label_list = [int(label) for label in label_list]

print("标签获取完成", flush=True)

# 加载分词工具
nlp = spacy.load('zh_core_web_md')

# 对文本进行分词
word_list = fet.split_word_from_sentence_array(nlp,text_list)
print("分词完成", flush=True)

vectorizer = TfidfVectorizer(max_features=3000)
vectorizer = vectorizer.fit_transform(word_list)

x_train, x_test,y_train, y_test = \
train_test_split(vectorizer,label_list,
                 test_size=0.2,random_state=0)
print("切分完成", flush=True)

x_train = x_train.toarray()
y_train = np.array(y_train)
x_test = x_test.toarray()
y_test = np.array(y_test)
end = time.time()

### studying with respect to Regularization parameter:

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

estimator= SVC(probability=True)
#parameters= {'C':[0.1,1,10,100,1000],'kernel':['polynomial', 'rbf', 'sigmoid','linear'],'gamma':[1,0.1,0.01,0.001]}
parameters= {'C':[0.1,1,10,100,1000]}
svc_clf= GridSearchCV(estimator=estimator,param_grid=parameters,cv=5,return_train_score=True,scoring='accuracy',verbose=10,error_score=0)
svc_clf.fit(x_train,y_train)
print(f'The best params are: {svc_clf.best_params_} ')
print(f'The best score are: {svc_clf.best_score_} ')

# from sklearn.externals import joblib
# joblib.dump(svc_clf,"model/svc.pkl")

import sklearn.metrics as sm

y_pred= svc_clf.predict(x_test)
print(f'The accuracy of train model after tuning is {sm.accuracy_score(y_train,svc_clf.predict(x_train))}')
print(f'The accuracy of test model after tuning is {sm.accuracy_score(y_test,y_pred)}')

import matplotlib.pyplot as plt
plt.figure(figsize=(50,24))
sm.plot_confusion_matrix(svc_clf,x_test,y_test)
plt.show()

from sklearn.metrics import confusion_matrix
# 计算预测结果和真实结果的混淆矩阵
cnf_matrix = confusion_matrix(y_test,y_pred)


label_str_lst = [
"婚恋交友",
"假冒身份",
"钓鱼网站",
"冒充公检法",
"平台诈骗",
"招聘兼职",
"杀猪盘",
"博彩赌博",
"信贷理财",
"刷单诈骗"]
from tool import evaluation_tool as elt
# 通过混淆矩阵计算准确率
accuracy_lst = elt.evaluate_accuracy(cnf_matrix)
callback_lst = elt.evaluate_callback(cnf_matrix)
elt.print_format_accuracy_and_callback(label_str_lst,cnf_matrix)


y_pred = svc_clf.predict_proba(x_test)
###################################
#          AUX绘图                #
##################################

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import LabelBinarizer

n_classes = 10

#标签改为独热编码
label_binarizer = LabelBinarizer().fit(y_test)
y_onehot_test = label_binarizer.transform(y_test)


# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# micro（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# macro（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('multi-calss ROC')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('./figs/auc.png')

# 计算pr

from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_onehot_test[:, i], y_pred[:, i])
    average_precision[i] = average_precision_score(y_onehot_test[:, i], y_pred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(
    y_onehot_test.ravel(), y_pred.ravel()
)
average_precision["micro"] = average_precision_score(y_onehot_test, y_pred, average="micro")

# setup plot details
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

_, ax = plt.subplots(figsize=(7, 8))

f_scores = np.linspace(0.2, 0.8, num=4)
lines, labels = [], []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

# add the legend for the iso-f1 curves
handles, labels = display.ax_.get_legend_handles_labels()
handles.extend([l])
labels.extend(["iso-f1 curves"])
# set the legend and the axes
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Extension of Precision-Recall curve to multi-class")

#plt.show()
plt.savefig('./figs/pr.png')