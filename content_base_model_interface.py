import spacy
from tool import feature_extraction_tool as fet
from spacy.lang.zh.stop_words import STOP_WORDS
from keras.models import Sequential
import keras
import numpy as np

nlp:spacy.Language

def content_based_model_init():
    global nlp
    # 加载分词工具
    nlp = spacy.load('zh_core_web_md')
    

def get_content_based_model_predict_vector(content:str) -> 'np.ndarray | None': 
    """基于内容的分类模型接口

    Args:
        content (str): 输入文本

    Returns:
        np.ndarray: 返回对所有分类的预测向量
        None: 输入不合法（字符不多于200词）

    """
    global nlp
    #句子转换为单词集
    wordSet = fet.split_to_word_set_from_sentence(nlp,content)
    #剔除常用词
    wordSet = fet.word_set_throw_stop_word(wordSet,STOP_WORDS)
    #归一化
    wordSet = fet.normalization_word_number(wordSet,number=200)
    #如果数量小于200，返回NONE
    if(len(wordSet) == 0):
        return None
    #字列表转换为矩阵
    wordSetVector = fet.wordSet_to_Matrix(nlp,wordSet)
    #加载模型
    model:Sequential = keras.models.load_model("./model/model_6_20_200_0411")
    #开始预测
    predict = model.predict(wordSetVector)
    #结果存储在predict
    return predict

def content_based_model_predict(content:str) -> int:
    """返回预测的标签对应的标号(0-12)

    Args:
        content (str): 输入文本

    Returns:
        int: 预测的标签对应的标号
        None: 输入不合法（字符不多于200词）
    """
    vec = get_content_based_model_predict_vector(content)
    if(vec is not None):
        return np.argmax(vec)
    else:
        return vec

string = "金融信息抵押贷款行业资讯机构理财协议存款房地产贷款最新新闻郑州银行商贸网络社区为您提供最新行业资讯金融信息最新新闻等栏目信息最具影响的协议存款机构理财抵押贷款房地产贷款等信息分享是提供信息最快最全的郑州银行商贸金融网站协议存款机构房地产郑州银行商贸网络社区首页最新新闻行业资讯金融信息当前位置首页什么是超跌次新股投资者如何选择超跌次新股真银行假行长美的集团遭遇十亿理财骗局没想到唱双簧的骗子竟是东方基金管理有限公司怡安翰威特咨询中信建投研究所等公司招聘信息尽在实习日报武汉公积金贷款实施新政首套房贷款额度为万银行卡怎么申请如何办理银行卡房产二次抵押唤醒自己沉睡的财产房产二次抵押唤醒自己沉睡的财产昆明市地方税务局官网如何登录承红色基因共书改革答卷召开复转军人座谈会嘉实基金官网首页怎样登录其将与鹏欣集团宣布战略合作剑指盘活存量资产公积金买房贷款流程与条件基金转换费率转换基金需要支出哪些费用真银行假行长美的集团遭遇十亿理财骗局没想到唱双簧的骗子竟是东方基金管理有限公司怡安翰威特咨询中信建投研究所等公司招聘信息尽在实习日报什么是超跌次新股投资者如何选择超跌次新股真银行假行长美的集团遭遇十亿理财骗局没想到唱双簧的骗子竟是东方基金管理有限公司怡安翰威特咨询中信建投研究所等公司招聘信息尽在实习日报武汉公积金贷款实施新政首套房贷款额度为万银行卡怎么申请如何办理银行卡更多资讯房贷将呈三大趋势再不看就晚了房贷将呈三大趋势再不看就晚了欧洲盘前高息货币继续表现强劲焦点转向欧盟二次峰会贷款按揭的房子还能贷款吗婚前协议赠与房产约定怎样才算有效美元瑞郎接近日均线阻力逾期不还后什么样的网贷会起诉我更多资讯欧元美元月日订单簿通乾证券投资基金是否具有投资价值信息放送融通基金女掌门率先推出协同事业部个人银行理财规划步骤是怎样个人所得税起征点是多少你知道吗个人所得税起征点调整后你能省下不少钱美国贷款买房指南看广告赚外快消费返利挣钱投资理财警惕十大陷阱沈阳全面取消房屋限购随手记理财记账电脑版下载后如何使用更多资讯版权所有时讯商贸金融网"
content_based_model_init()
predict  = content_based_model_predict(string)
print(predict)