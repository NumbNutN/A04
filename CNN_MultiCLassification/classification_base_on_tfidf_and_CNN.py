from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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

for i in [0,1,2]:
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:20],
                                col = 1))
    
    # 由于kears要求使用数字作为标签
    label_list.extend(ct.read_csv_label_a2i(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:20]
                                ))
for i in [10,12,15]:
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:20],
                                col = 1
                                ))
    
    label_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:20],
                                col = 2
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

# 数据测试集切分
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = \
train_test_split(vectorizer,label_list,
                 test_size=0.2,random_state=0)
print("切分完成", flush=True)

x_train = x_train.toarray()
y_train = np.array(y_train)
x_test = x_test.toarray()
y_test = np.array(y_test)
end = time.time()

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense,Dropout
from keras.regularizers import l2

# 定义数据的输入形状
input_shape = (1,x_train.shape[1])

# 定义数据分类的类的数量
num_classes = 12

# 定义CNN模型
model = Sequential()

# 添加一个1D卷积层，有32个过滤器，核大小为3，并有relu激活函数
# 添加一个MaxPooling1D层，池大小为2
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)))
model.add(MaxPooling1D(pool_size=2))

# 添加一个Flatten层，将前一层的输出转换为一维矢量
model.add(Flatten())

# 添加一个具有64个单元和relu激活函数的全连接密集层
model.add(Dense(units=64, activation='relu'))

# 添加一个具有softmax激活函数的输出层，将我们的数据分类为num_classes
model.add(Dense(units=3, activation='softmax'))

# 用分类交叉熵损失函数和adam优化器来编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall', 'f1_score'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#训练
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

#保存模型
model.save("./model_6_200_0405")


# 对测试集预测得到预测结果
y_pred = model.predict(x_test)

from keras.metrics import accuracy
# 计算损失和准确率
scores = model.evaluate(x_test,y_test)

#打印损失和准确率
print(scores)


# 对测试集预测得到预测结果
y_pred = model.predict(x_test)

from keras.metrics import accuracy
# 计算损失和准确率
scores = model.evaluate(x_test,y_test)

#打印损失和准确率
print(scores)