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

def calculate_score(label_list:'list[int]',y_pred:'np.ndarray',n_classes:int,class_list:'list[str]'):
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.metrics import precision_recall_curve
    #标签改为独热编码
    label_binarizer = LabelBinarizer().fit(label_list)
    y_onehot_test = label_binarizer.transform(label_list)

    # For each class
    precision = dict()
    recall = dict()
    threshold = dict()

    for i in range(n_classes):
        precision[i], recall[i], threshold = precision_recall_curve(y_onehot_test[:, i], y_pred[:, i])


    precision_at_recall_thresholds_all = []
    precision_at_recall_thresholds = []
    for i in range(n_classes):
        for threshold in [0.7,0.8,0.9]:
            cloesst_recall_index = np.argmin(np.abs(recall[i]-threshold))
            precision_at_recall_thresholds.append(precision[i][cloesst_recall_index])
        precision_at_recall_thresholds_all.append(precision_at_recall_thresholds)
        precision_at_recall_thresholds = []

    for i in range(n_classes):
        scores = 0.5* precision_at_recall_thresholds_all[i][0] + 0.3*precision_at_recall_thresholds_all[i][1] + 0.2*precision_at_recall_thresholds_all[i][2]
        print(class_list[i]+" scores is"+str(scores))

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import LabelBinarizer

from matplotlib import font_manager

colors = cycle(['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881'])

def draw_init():
    #font_manager.FontManager().addfont(path="/usr/share/fonts/truetype/dejavu/Dengb.ttf")
    #fontmanager.addfont(path="/usr/share/fonts/truetype/dejavu/Dengb.ttf")
    plt.rcParams["font.sans-serif"]=["DengXian"] #设置字体
    plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题    

def draw_roc(imgPath:str,test_labels:'list[int]',y_pred:'list',n_classes:int,class_list:'list[str]'):
    # plt.rc("font",family='DengXian')
    # plt.rcParams["font.sans-serif"]=["DengXian"] #设置字体
    # plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
    #标签改为独热编码
    label_binarizer = LabelBinarizer().fit(test_labels)
    y_onehot_test = label_binarizer.transform(test_labels)


    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class' + class_list[i] +' (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(imgPath)


from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import font_manager
def draw_pr(imgPath:str,test_labels:'list[int]',y_pred:'list',n_classes:int,class_list:'list[str]'):
    # plt.rc("font",family='DengXian')
    # plt.rcParams["font.sans-serif"]=["DengXian"] #设置字体
    # plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
    #标签改为独热编码
    label_binarizer = LabelBinarizer().fit(test_labels)
    y_onehot_test = label_binarizer.transform(test_labels)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_onehot_test[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_onehot_test[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_onehot_test.ravel(), y_pred.ravel()
    )
    average_precision["micro"] = average_precision_score(y_onehot_test, y_pred, average="micro")

    # setup plot details

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class" + class_list[i], color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    #plt.show()
    plt.savefig(imgPath)


import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,filePath:str,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵图形
    :param cm: 混淆矩阵
    :param classes: 标签名称列表
    :param normalize: 是否对混淆矩阵进行归一化
    :param title: 图表标题
    :param cmap: 颜色映射
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.tight_layout()
    plt.savefig(filePath)