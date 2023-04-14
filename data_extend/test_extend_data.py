
import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct

import numpy as np
import csv

####################################################################################
#                                汇总增强数据集                                      #
####################################################################################


dfl = fet.DataFeature()
text_list = []
label_list = []
content_list = []
# 3分类
# 字符串标签转数字
from tool import classification_tool as ct

label_list.extend(fet.read_csv_context(
                                filename="./data/all_content.csv",
                                row_range = range(60000),
                                col = 1))

content_list.extend(fet.read_csv_context(
                                filename="./data/all_content.csv",
                                row_range = range(60000),
                                col = 0))


info_list = []

oldlabel:str = ''
oldcnt:int=0
idx:int=0
for label in label_list:
    if label != oldlabel:
        info_list.append({'label':oldlabel,'count':oldcnt,'start':idx-oldcnt,'end':idx})
        oldcnt=0
        oldlabel=label
    oldcnt+=1
    idx+=1
info_list.append({'label':oldlabel,'count':oldcnt,'start':idx-oldcnt,'end':idx})
pass