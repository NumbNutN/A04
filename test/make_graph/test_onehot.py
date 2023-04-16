from sklearn.preprocessing import LabelBinarizer

#测试集标签
y_test = []
#测试集预测值
y_pred = []

#标签改为独热编码
label_binarizer = LabelBinarizer().fit(y_test)
y_onehot_test = label_binarizer.transform(y_test)

