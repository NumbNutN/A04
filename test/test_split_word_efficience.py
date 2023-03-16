import sys
sys.path.append(".")

from tool import feature_extraction_tool as fet
import spacy
import time

# 加载网页文本快照数据 train.csv
filename = "./data/train2.csv"
# 数据集范围 定义从csv文件中提取的数据的范围
data_range = range(650,800)

text_list = fet.read_csv_context(filename,data_range)

# 加载分词工具
nlp = spacy.load('zh_core_web_md')

start_time = time.time()
# 对文本进行分词
word_list = fet.split_word_arr(nlp,text_list)

end_time = time.time()

print("用时",end_time-start_time)

lst:list = fet.split_word_sentence_to_split_word_list(word_list[0])

print(lst)