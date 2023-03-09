####################################################################################
#                                特征提取工具组                                     #
#  -- 用于训练测试数据预处理，特征工程等                                              #
#  -- 通过 import classification_tool as cft 使用 cft.func                          #
####################################################################################
import spacy
import csv


def read_csv_context(filename:str,row_range:range,col:int = 1) -> list:
    """读取csv文件并依据指定的行数范围和列数将文本装载在列表中返回
        注意由于数据集的csv文件网页快照文本往往在第2列，
        col参数默认为第二列

    Args:
        filename (str): csv文件的路径
        row_range (range): 行数范围
        col (int): 列数，单个数据

    Returns:
        list: 装载文本的列表

    Example:
        read_csv_context("xx.csv",range(1,4),1)
        return:csv文件的第2,3,4排的第2列将作为数组的元素
        ["row2 col2","row3 col2","row4 col2"]

    Info:
        Created by LGD on 2023-3-9
        Last update on 2023-3-9
    
    """
    data = []
    cnt = 0
    with open(filename,encoding="UTF8") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if cnt in row_range:
                data.append(row[col])
            cnt += 1
    return data



def split_word(nlp:spacy.Language,sentence:str,split_word:str = ' ') -> str:
    """将一个句子以split_word为间隔进行分词

    Args:
        nlp (spacy.Language): spacy Language模型
        sentence (str): 未分词的中文
        split_word (str, optional): 分割词符，默认为空格. Defaults to ' '.

    Returns:
        str: 分词后的字符串

    Example:
        split_word(nlp,"唤醒自己沉睡的财产房产")
        return:"唤醒 自己 沉睡 的 财产 房产"

    Info:
        Created by LGD on 2023-3-9
        Last update on 2023-3-9
    """
    words = nlp(sentence)
    lst = [str(word) for word in words]
    #word 的数据类型为 <class 'spacy.tokens.token.Token'>

    return split_word.join(word for word in lst)



def split_word_arr(nlp:spacy.Language,sentence_arr:list,split_word:str = ' ') -> list:
    """将一个句子以split_word为间隔进行分词

    Args:
        nlp (spacy.Language): spacy Language模型
        sentence (list): 未分词的中文
        split_word (str, optional): 分割词符，默认为空格. Defaults to ' '.

    Returns:
        str: 分词后的字符串

    Example:
        split_word_arr(nlp,["唤醒自己沉睡的财产房产","骗子竟是东方基金"])
        return:["唤醒 自己 沉睡 的 财产 房产","骗子 竟是 东方 基金"]

    Info:
        Created by LGD on 2023-3-9
        Last update on 2023-3-9
    """
    sentence_list = []
    for sentence in sentence_arr:
        words = nlp(sentence)
        single_sentence_lst = [str(word) for word in words]
        sentence_list.append(split_word.join(word for word in single_sentence_lst))
        
    return sentence_list




