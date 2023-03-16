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
# 四分类
for i in range(0,4):
    text_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range = dfl.dataFeatureList[i]["range"][0:50],
                                col = 1))
    label_list.extend(fet.read_csv_context(
                                filename="./data/"+dfl.dataFeatureList[i]["fileName"],
                                row_range =dfl.dataFeatureList[i]["range"][0:50],
                                col = 2))
    
# 加载分词工具
nlp = spacy.load('zh_core_web_md')

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

# 将单词列表转化为词向量
word_gather_vec = fet.word_gather_arr_to_vec(nlp,word_set_list)

print("词向量转换完成")
print(
    "词向量矩阵类型:"+str(type(word_gather_vec[0])),
    "列表类型"+str(type(word_gather_vec))
    )
for i in range(len(word_gather_vec)):
    print("矩阵维度:",
            i,
            word_gather_vec[i].shape
        )
# 低于600词的可以被剔除










