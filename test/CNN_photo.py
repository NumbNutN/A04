from keras.models import Model
from keras.layers import Input, Dense, concatenate, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Embedding


####################################################################################
#                                     数据选择                                      #
####################################################################################
from tool import feature_extraction_tool as fet
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
import cupy

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
                                row_range = dfl.dataFeatureList[i]["range"][0:5],
                                col = 1))
    
    # 由于kears要求使用数字作为标签
    label_list.extend(ct.get_label_from_csv(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:5]
                                ))
# 加载分词工具
nlp = spacy.load('zh_core_web_md')
word_set_list = fet.split_to_word_set_from_sentence(nlp,text_list)
print("分词完成")

#去除停用词
from spacy.lang.zh.stop_words import STOP_WORDS

fet.word_set_throw_stop_word(word_set_list,list(STOP_WORDS))
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
word_gather_vec = fet.wordSet_to_Matrix(nlp,word_set_list,is_flat=True)
end_word2vec = time.time()
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
    
x_train, x_test,y_train, y_test = \
train_test_split(word_gather_vec,label_list,
                 test_size=0.2,random_state=0)
print("切分完成", flush=True)

#转换为CPU类型
#2023-3-24
#这是因为tensorflow仍然采用了CPU运算
x_train = cupy.asnumpy(x_train)
x_test = cupy.asnumpy(x_test)

#定义数据特征工程
max_len = 180000


# 定义图像输入
image_input = Input(shape=(img_width, img_height, num_channels), name='image_input')
#两次卷积和池化
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#将输出平铺成一个一维数组，下同
flatten1 = Flatten()(pool2)

# 定义文本输入
text_input = Input(shape=(max_len,), name='text_input')
#将每个单词索引映射到大小为embedding_dim的密集向量，input_dim是词汇表的大小，output_dim是密集嵌入向量的大小，input_length是每个输入序列的长度。
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(text_input)
flatten2 = Flatten()(embedding)

# 联合图像和文本输入
merged = concatenate([flatten1, flatten2], axis=-1)

# 为分类添加全连接层
fc1 = Dense(128, activation='relu')(merged)
fc2 = Dense(num_classes, activation='softmax')(fc1)

# 定义复合模型
model = Model(inputs=[image_input, text_input], outputs=fc2)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#train_text和train_images是模型的输入数据，其中train_text代表文本数据，train_images代表图像数据。 train_labels是输入数据的相应输出数据或标签。
#epochs参数指定了模型在整个训练数据集上训练的次数。batch_size参数指定了每次训练迭代中使用的样本数。
model.fit([train_text, train_images], train_labels, epochs=10, batch_size=32)

history = model.fit([train_text, train_images], train_labels, epochs=10, batch_size=32)
#train_loss和val_loss是分别包含模型在每个历时的训练和验证损失的列表
train_loss = history.history['loss']
val_loss = history.history['val_loss']

#卷积层和集合层通常用于图像处理任务，因为它们能有效地捕捉输入图像的局部模式和结构。图像通常有一个空间结构，像素排列在一个网格中，
# 卷积层的设计是为了通过学习局部过滤器来利用这一结构，这些过滤器可以检测到诸如边缘、角落和纹理的模式。
#池化层被用来对卷积层产生的特征图进行降样，这就减少了数据的空间尺寸，同时保留了最突出的特征。这有助于减少网络所需的计算量，也有助于防止过度拟合。
#相比之下，文本数据没有空间结构，所以卷积层和池化层并不适合捕捉文本数据的模式和结构。相反，嵌入层通常被用来将文本数据转化为可以输入神经网络的数字表示。
#嵌入层将文本中的每个独特的词映射到连续矢量空间中的一个密集矢量。
#这使得神经网络能够根据词语在向量空间中的接近程度来学习它们之间的有意义的关系。
#通过对文本数据使用嵌入层，对图像数据使用卷积层和池化层，我们可以利用每种类型的层的优势，从输入数据中提取有意义的特征。