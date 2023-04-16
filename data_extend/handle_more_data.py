import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import spacy
import numpy as np
import csv
csv.field_size_limit(20000*6000*10)
####################################################################################
#                                     数据选择                                      #
####################################################################################


dfl = fet.DataFeature()
spacy.prefer_gpu()
text_list = []
label_list = []
content_list = []
# 3分类
# 字符串标签转数字
from tool import classification_tool as ct



text_list.extend(fet.read_csv_context(
                                filename="./data/train1.csv",
                                row_range = range(3000000),
                                col = 0))
    
label_list.extend(fet.read_csv_context(
                                filename="./data/train1.csv",
                                row_range = range(30000),
                                col = 1))

# content_list.extend(fet.read_csv_context(
#                                 filename="./data/train1.csv",
#                                 row_range = range(3000000),
#                                 col = 1))



#     # 由于kears要求使用数字作为标签
# label_list.extend(ct.get_label_from_csv(
#                                 filename="./data/"+dfl.dataFeatureList[6]["fileName"],
#                                 row_range =dfl.dataFeatureList[6]["range"]
#                                 ))


# with open("./data/more_haveContent0402.csv",'w',encoding='utf-8') as csvFile:
#     writer = csv.writer(csvFile)
#     for i in range(len(content_list)):
#         if content_list[i] != "Connect Failed" and content_list[i] != "Nothing":
#             writer.writerow([text_list[i],content_list[i],label_list[i]])

####################################################################################
#                                     写入数据                                      #
####################################################################################

# import pandas as pd


# idx:int = 0
# for i in range(len(content_list)):
#     if content_list[idx] == "Connect Failed" or content_list[idx] == "Nothing":
#         text_list.pop(idx)
#         # label_list.pop(idx)
#         content_list.pop(idx)
#     else:
#         idx+=1

# data = {
#     'url':text_list,
#     'text':content_list,
#     # 'label':[int(label) for label in label_list]
# }

# df = pd.DataFrame(data)
# df = df.sort_values(by='label',ascending=True)
# df.to_csv("./data/test_0413.csv",sep=',',mode='w',header=False,index=False,encoding='utf-8')


####################################################################################
#                                     数据统计                                      #
####################################################################################

cnt = 0
#统计正常标签
for label in label_list:
    if label == '0':
        cnt += 1

# 统计一些数据
cnt_cntfail = 0
cnt =0
cnt_nothing = 0
for i in range(len(content_list)):
    if content_list[i] != "Connect Failed" and content_list[i] != "Nothing":
        cnt+=1
    if content_list[i] == "Connect Failed":
        cnt_cntfail+=1
    if content_list[i] =="Nothing":
        cnt_nothing+=1

print("网址总数 %d" %(len(content_list)))
print("有文本的数量 %d" %(cnt))
print("Connect Failed数量 %d" %(cnt_cntfail))
print("Nothing数量 %d" %(cnt_nothing))
print("有文本的比例：%f" %(cnt / len(content_list)))
print("Connect Failed比例：%f" %(cnt_cntfail / len(content_list)))
print("Nothing比例：%f" %(cnt_nothing / len(content_list)))


####################################################################################
#                                     统计段首                                      #
####################################################################################



# label_list.extend(fet.read_csv_context(
#                                 filename="./data/"+dfl.dataFeatureList[8]["fileName"],
#                                 row_range = dfl.dataFeatureList[8]["range"],
#                                 col = 2))

# info_list = []

# oldlabel:str = ''
# oldcnt:int=0
# idx:int=0
# for label in label_list:
#     if label != oldlabel:
#         info_list.append({'label':oldlabel,'count':oldcnt,'start':idx-oldcnt,'end':idx})
#         oldcnt=0
#         oldlabel=label
#     oldcnt+=1
#     idx+=1


pass

