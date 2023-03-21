import tensorflow as tf

import sys
sys.path.append("..")
from tool import feature_extraction_tool as fet

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 128 # 词向量维度
    seq_length = 100  # 序列长度
    num_classes = 26  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 8000  # 词汇表大小

    hidden_dim = 256  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10 # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        # 强制代码在CPU上面执行操作。因为默认情况下，TensorFlow会尝试将操作放在GPU上面进行运行（如果存在GPU），
        # 但是嵌入层的操作目前还不支持GPU运行，所以如果你不指定CPU进行运行，那么程序会报错。
        with tf.device('/cpu:0'),tf.name_scope("Embedding"):
            self.embedding_var = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            # 选取embedding_var里边以self.input_x为索引的元素
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_var, self.input_x)

        with tf.name_scope("CNN"):
            # CNN layer
            conv = tf.layers.conv1d(self.embedding_inputs, self.config.num_filters, self.config.kernel_size, name='Convolution')
            # global max pooling layer降维
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("Score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='FC1')
            fc = tf.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='FC2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("Optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("Accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))