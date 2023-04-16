import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


plt.get_cmap()

filePath = "./final/label_and_pred.txt"

label_List = ["婚恋交友",
            "假冒身份",
            "钓鱼网站",
            "冒充公检法",
            "平台诈骗",
            "招聘兼职",
            "杀猪盘",
            "博彩赌博",
            "信贷理财",
            "刷单诈骗"]
from itertools import cycle
colors = cycle(['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881'])
'''
colors  = ['#C630C9', '#3855A5', '#E22C90', '#34B34A', '#45287A', '#D64673', '#E1DA6D',
            '#DB83E0', '#FFA1C4', '#8770E0', '#01AFEE', '#4574C6', '#FDC100', '#BAD0C4',
            '#474552', '#496571', '#39443E', '#A587A6', '#E0BDC9', '#F2E5DA', '#DAD9DD']
'''


with open(filePath,mode='r',encoding='utf-8') as fileObj:
    lines = fileObj.readlines()

lines.pop(0)
#解析文件
for idx in range(len(lines)):
    lines[idx] = lines[idx].strip('\n')
    lines[idx] = lines[idx].split('\t')
    lines[idx].pop(len(lines[idx])-1)

y_label = []

for idx in range(len(lines)):
    y_label.append(lines[idx].pop(0))

y_pred = lines

y_label = [int(label) for label in y_label]

for idx in range(len(y_pred)):
    y_pred[idx] = [float(elem) for elem in y_pred[idx]]

###################################
#          AUX绘图                #
##################################

import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import LabelBinarizer

n_classes = 10

#标签改为独热编码
label_binarizer = LabelBinarizer().fit(y_label)
y_onehot_test = label_binarizer.transform(y_label)

#预测值转numpy
y_pred = np.array(y_pred)


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
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
 
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
 
#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#colors = cycle(['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881'])

for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
            label='ROC curve of ' + label_List[i] + '(area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
 
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('multi-calss ROC')
plt.legend(loc="lower right")

plt.savefig('./final/figs/auc0-4.png')
#plt.show()

###################################
#          AUX绘图                #
##################################

import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import LabelBinarizer

n_classes = 10

#标签改为独热编码
label_binarizer = LabelBinarizer().fit(y_label)
y_onehot_test = label_binarizer.transform(y_label)

#预测值转numpy
y_pred = np.array(y_pred)


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
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
 
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
 
#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#colors = cycle(['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881'])

for i, color in zip(range(5,10), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
            label='ROC curve of ' + label_List[i] + '(area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
 
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('multi-calss ROC')
plt.legend(loc="lower right")

plt.savefig('./final/figs/auc5-9.png')
#plt.show()




###################################
#          PR绘图                #
##################################


from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle

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

#colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
#colors = cycle(['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881'])
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

for i, color in zip(range(5), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for class {label_List[i]}", color=color)

# add the legend for the iso-f1 curves
handles, labels = display.ax_.get_legend_handles_labels()
handles.extend([l])
labels.extend(["iso-f1 curves"])
# set the legend and the axes
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Extension of Precision-Recall curve to multi-class")

plt.savefig('./final/figs/pr0-4.png')
plt.show()

###################################
#          PR绘图                #
##################################


from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt
from itertools import cycle

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

#colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
#colors = cycle(['#3682be','#45a776','#f05326','#eed777','#334f65','#b3974e','#38cb7d','#ddae33','#844bb3','#93c555','#5f6694','#df3881'])
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

for i, color in zip(range(5,10), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for class {label_List[i]}", color=color)

# add the legend for the iso-f1 curves
handles, labels = display.ax_.get_legend_handles_labels()
handles.extend([l])
labels.extend(["iso-f1 curves"])
# set the legend and the axes
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Extension of Precision-Recall curve to multi-class")

plt.savefig('./final/figs/pr5-9.png')
plt.show()