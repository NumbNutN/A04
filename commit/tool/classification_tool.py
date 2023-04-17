####################################################################################
#                                分类模型工具组                                     #
#  -- 用于输入数据的分类任务                                                         #
#  -- 通过 import feature_extraction_tool as fet 使用方法  fet.func                 #
####################################################################################

import spacy
import csv
from tool import feature_extraction_tool as fet

class label:
    label_dict = {"正常":0,
                  "购物消费":1,
                  "婚恋交友":2,
                  "假冒身份":3,
                  "钓鱼网站":4,
                  "冒充公检法":5,
                  "平台诈骗":6,
                  "招聘兼职":7,
                  "杀猪盘":8,
                  "博彩赌博":9,
                  "信贷理财":10,
                  "刷单诈骗":11,
                  "中奖诈骗":12}
    
    def get_label(str_label:str)->int:
        """根据字符串获得标签序号

        Args:
            str_label (str): 字符串型标签

        Returns:
            int: 标签序号

        Info:
            Created by LGD on 2023-3-9
            Last update on 2023-3-9
        """
        return label.label_dict[str_label]

def get_label_from_csv(filename:str,row_range:range) -> list:
    """从csv依据指定行数获取标签序号
        "正常":0,
        "购物消费":1,
        "婚恋交友":2,
        "假冒身份":3,
        "钓鱼网站":4,
        "冒充公检法":5,
        "平台诈骗":6,
        "招聘兼职":7,
        "杀猪盘":8,
        "博彩赌博":9,
        "信贷理财":10,
        "刷单诈骗":11,
        "中奖诈骗":12

    Args:
        filename (str): 获取标签的csv文件来源
        row_range (range): 数据条目范围（行数）

    Returns:
        list: 以条目为顺序的标签的列表

    Info:
        Created by LGD on 2023-3-9
        Last update on 2023-3-9
    """
    label_list_str = fet.read_csv_context(filename,row_range,2)
    label_list_int = [label.get_label(label_str) for label_str in label_list_str]
    return label_list_int
