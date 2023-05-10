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

estimator= SVC()
#parameters= {'C':[0.1,1,10,100,1000],'kernel':['polynomial', 'rbf', 'sigmoid','linear'],'gamma':[1,0.1,0.01,0.001]}
parameters= {'C':[0.1,1,10,100,1000]}
svc_clf= GridSearchCV(estimator=estimator,param_grid=parameters,cv=5,return_train_score=True,scoring='accuracy',verbose=10,error_score=0)
svc_clf.fit(x_train,y_train)
print(f'The best params are: {svc_clf.best_params_} ')
print(f'The best score are: {svc_clf.best_score_} ')


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
"刷单诈骗",
"中奖诈骗"]
from tool import evaluation_tool as elt
# 通过混淆矩阵计算准确率
accuracy_lst = elt.evaluate_accuracy(cnf_matrix)
callback_lst = elt.evaluate_callback(cnf_matrix)
elt.print_format_accuracy_and_callback(label_str_lst,cnf_matrix)