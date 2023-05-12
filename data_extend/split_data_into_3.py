import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
from tool import classification_tool as ct
import spacy
import numpy as np
import csv

text_list = (fet.read_csv_context(
                                filename="./data/all_content_without_split.csv",
                                row_range = range(3000000),
                                col = 0))

label_list = (fet.read_csv_context(
                                filename="./data/all_content_without_split.csv",
                                row_range = range(3000000),
                                col = 1))

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = \
train_test_split(text_list,label_list,
                 test_size=0.2,random_state=0)

import pandas as pd
data = {
    'text':x_train,
    'label':[int(label) for label in y_train]
}
df = pd.DataFrame(data)
df = df.sort_values(by='label',ascending=True)
df.to_csv("./data/all_content_without_split_train.csv",sep=',',mode='w',header=False,index=False,encoding='utf-8')

data = {
    'text':x_test,
    'label':[int(label) for label in y_test]
}
df = pd.DataFrame(data)
df = df.sort_values(by='label',ascending=True)
df.to_csv("./data/all_content_without_split_test.csv",sep=',',mode='w',header=False,index=False,encoding='utf-8')