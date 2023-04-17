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

import numpy as np
y_pred = np.array(y_pred)

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_curve
#标签改为独热编码
label_binarizer = LabelBinarizer().fit(y_label)
y_onehot_test = label_binarizer.transform(y_label)

n_classes = 10

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



# precison, recall, threshold = precision_recall_curve(y_label,y_pred)

for i in range(n_classes):
    scores = 0.5* precision_at_recall_thresholds_all[i][0] + 0.3*precision_at_recall_thresholds_all[i][1] + 0.2*precision_at_recall_thresholds_all[i][2]
    print(label_List[i]+" scores is"+str(scores))
pass

