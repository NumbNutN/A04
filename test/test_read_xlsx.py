####################################################################################
#                                   测试读取xlsx                                    #
#  -- date:2023-3-14                                                               #
####################################################################################

import sys
sys.path.append(".")

# 特征处理工具库
from tool import feature_extraction_tool as fet

dataPath = "./data/train3.xlsx"
print(fet.read_xlsx_context(dataPath,range(1,3),0))
print(type(fet.read_xlsx_context(dataPath,range(1,3),0)))





