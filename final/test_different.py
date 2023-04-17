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
url2_list = []

csv.field_size_limit(20000*6000)


url_list.extend(fet.read_csv_context(
                                filename="./final/data/test_with_content_all.csv",
                                row_range = range(1,11147),
                                col = 0))
url2_list.extend(fet.read_csv_context(
                                filename="./final/data/test_all_all_pred_0417.csv",
                                row_range = range(0,11142),
                                col = 0))

for i in range(len(url_list)):
    if(url_list[i] != url2_list[i]):
        pass

# for url1, url2 in url_list,url2_list:
#     if(url1!=url2):
#         pass
