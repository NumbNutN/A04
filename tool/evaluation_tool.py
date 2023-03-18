####################################################################################
#                                模型评估工具库                                     #
#  -- 用于模型效果的量化，如准确率、召回率的计算等                                     #
#  -- 通过 import evaluation_tool as elt 使用 elt.func                              #
####################################################################################

import numpy as np

def evaluate_accuracy(matrix:np.ndarray) -> list:
    """评估模型准确率（单一标签项）

    Args:
        matrix (np.ndarray): 混淆矩阵

    Returns:
        list: 以矩阵顺序返回模型的准确率
    """
    accuracy_lst = []
    cnt_tp_and_fp:int = 0
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            cnt_tp_and_fp += matrix[i][j]
        accuracy_lst.append(matrix[j][j] / cnt_tp_and_fp)
        cnt_tp_and_fp = 0
    
    return accuracy_lst

def evaluate_callback(matrix:np.ndarray) -> list:
    """评估模型召回率（单一标签项）

    Args:
        matrix (np.ndarray): 混淆矩阵

    Returns:
        list: 以矩阵顺序返回模型的准确率
    """
    call_lst = []
    cnt_tp_and_fp:int = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cnt_tp_and_fp += matrix[i][j]
        call_lst.append(matrix[i][i] / cnt_tp_and_fp)
        cnt_tp_and_fp = 0
    
    return call_lst

def print_format_accuracy_and_callback(label_lst:list,matrix:np.ndarray) -> None:
    """格式化输出各个标签的准确率和召回率

    Args:
        label_lst (list): 和混淆矩阵纵轴一致的标签列表
        matrix (np.ndarray): 混淆矩阵
    """

    ac_lst = evaluate_accuracy(matrix)
    call_lst = evaluate_callback(matrix)

    print("%-10s%-10s%-10s" %("标签","准确率","召回率"))
    for i in range(min(len(ac_lst),len(call_lst))):
        print("%-10s%-10f%-10f" %(label_lst[i],ac_lst[i],call_lst[i]))
        


