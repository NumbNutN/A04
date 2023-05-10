import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import numpy as np
import csv
import spacy
import copy

dfl = fet.DataFeature()

spacy.prefer_gpu()

url_list = []
label_list = []

csv.field_size_limit(20000*6000)


url_list.extend(fet.read_csv_context(
                                filename="./final/data/test_all_all_pred_0417.csv",
                                row_range = range(0,20000),
                                col = 0))

url_list.extend(fet.read_csv_context(
                                filename="./final/data/normal_all_all_pred_0417.csv",
                                row_range = range(0,2000000),
                                col = 0))

label_list.extend(fet.read_csv_context(
                                filename="./final/data/test_all_all_pred_0417.csv",
                                row_range = range(0,20000),
                                col = 1))

label_list.extend(fet.read_csv_context(
                                filename="./final/data/normal_all_all_pred_0417.csv",
                                row_range = range(0,2000000),
                                col = 1))



import pandas as pd
data = {
    'url':url_list,
    'label':label_list
}

df = pd.DataFrame(data)
df = df.sort_values(by='url',ascending=True)
df.to_csv("./final/data/normal_and_cheat_pred_all_0417.csv",sep=',',mode='w',header=False,index=False,encoding='utf-8')

