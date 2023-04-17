########################################################
#包：
#
#re:用于通过正则表达式筛选文本
#time:用以命名
#seleninum：用以打开网站截图
#PIL：打开截图
#pytesseract：分析图片提取文本
#requests:请求网址并返回加载信息
#chardet：检查返回的语言的编码信息
#pandas：读取和写csv文件
#bs4：获取网站html语言中的文本信息
#
#
#注：pytesseract需要先将pytesseract.py文件的tesseract_cmd = 'tesseract.exe'改为
#tesseract_cmd = r'tesseract.exe对应路径'
#
#以上的包均可在python3.7下使用
#如果不打算使用截图，那么只需要删除seleninum，PIL，pytesseract和get_photo(),get_textofphoto()即可通过编译执行
#################################################################
#函数及其用处：
#timeout_record(url)：将网址写入text文档用于记录
#confail_record(url):无法与网址建立连接，记录
#con_try(url):利用requess库ping一下，返回statu
#check(url):检查网址格式规范，防止因格式问题导致出现问题
#get_textofhtml_train(url):通过分析网页html获得文本信息
#get_textofhtml_test(url):作用同上，区别见下具体代码注释
#get_photo(url,path)：获取网站截图
#get_textofphoto(path):获取先前捕获的截图中文字
#read_csv(csv_path, name):读取csv文件，获取对应name列并转换成列表
#gettext(**kwargs):一定要使用关键字输入参数，输入url，则为测试时获取文本，输入csv文件路径时，则是获得训练集下面为具体使用例子:
#gettext(url= 'www.baidu.com')
#gettext(csv_path= 'train1.csv')
###############################################################
import re
import time
import chardet
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
import requests
from requests.exceptions import ConnectionError
from PIL import Image
import pytesseract
import pandas as pd
from bs4 import BeautifulSoup

###############################################################################测试所用变量
path = r"C:\Users\13360\Desktop\scrrenshot"
PathOfChrome = r"C:\Users\13360\Desktop\chromedriver_win32\chromedriver.exe"
c_path = r"C:\Users\13360\Desktop\train3.csv"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
prefix = 'http://'
################################################################################

#全局变量，用于分析
#exa:时间戳暂存变量
#cap_f:标志是否捕获成功
global exa
global cap_f
##################################
#设置截屏使用的浏览器
#PathOfChrome是驱动绝对路径，此处使用Chrome
####################
#driver = webdriver.Chrome(PathOfChrome)
####################

######################加载超时记录


def timeout_record(url):
    with open('timeout.txt', 'a') as f:
        f.write(url)
        f.write("\n")
#######################连接失败记录


def confail_record(url):
    with open('connect failed.txt', 'a') as f:
        f.write(url)
        f.write("\n")
##################尝试连接


def con_try(url):
    global headers
    try:
        respense = requests.get(url, headers=headers)
        return respense.status_code
    except ConnectionError:
        confail_record(url)

#####################检查url格式规范


def check(url):
    global prefix
    if url[0:len(prefix)] != prefix:
        url = prefix + url
    return url

##########################################################################################
#                   通过html获取文本（二者的区别在于对异常的处理和返回值，一个用于训练前，一个用于测试时）
##########################################################################################


def get_textofhtml_train(url):
    url_ = check(url)
    try:
        responese = requests.get(url_, headers=headers, timeout=5)
        encoding = chardet.detect(responese.content)['encoding']

        soup = BeautifulSoup(responese.content.decode(encoding=encoding), 'html.parser')
        text = re.sub(r'[^\u4e00-\u9fa5]', '', soup.get_text())
        text = text.replace('\n', '')
        return text
    except:
        confail_record(url)
        return 'Nothing'


def get_textofhtml_test(url):
    try:
        responese = requests.get(url, headers=headers, timeout=5)
        encoding = chardet.detect(responese.content)['encoding']

        soup = BeautifulSoup(responese.content.decode(encoding=encoding), 'html.parser')
        text = re.sub(r'[^\u4e00-\u9fa5]', '', soup.get_text())
        text = text.replace('\n', '')
        flag = True
        return text,flag
    except:
        flag = False
        return 'Nothing',flag


##################################################################
#                   从单个网站中截屏，保存，访问失败的网站写入text文档
##################################################################

def get_photo(url,path):#url为网址，不必写全http。path为存放截图照片的绝对路径
    try:
        global exa
        global cap_f
        exa = time.time()
        driver = webdriver.Chrome(PathOfChrome)
        driver.maximize_window()
        driver.set_page_load_timeout(3)
        driver.get(url)
        po_path = path + "\\" + "%f.png" % exa
        driver.get_screenshot_as_file(po_path)#采用时间戳命名，可以随截随分析
        cap_f = True
    except TimeoutException:
        timeout_record(url)
        cap_f = False
    except WebDriverException:
        confail_record(url)
        cap_f = False
    finally:
        driver.quit()

##################################################################
#               输入图片路径，获得图片文字
##################################################################

def get_textofphoto(path):
    global exa
    po_path = path + "\\" + "%f.png" % exa
    image = Image.open(po_path)
    text = pytesseract.image_to_string(image, lang='chi_sim')
    text = text.replace("“","").replace("。","").replace(" ","").replace("\n","")
    text = re.sub(r'\d+', '', text)
    text = re.sub('[a-zA-Z]', '',text)
    return text

#################################################################
#               读取csv文件对应列，并转换成列表
#################################################################

def read_csv(csv_path, name):
    data = pd.read_csv(csv_path)
    data = data[name].tolist()
    return data


def gettext(**kwargs):
    if 'url' in kwargs and 'csv_path' not in kwargs:
        url = kwargs["url"]
        url_ = check(url)
        text,flag = get_textofhtml_test(url_)
        if flag == True:
            return text
        else:
            #访问失败需要记录可添加语句
            return url
    elif 'csv_path' in kwargs:
        csv_path = kwargs["csv_path"]
        urls = read_csv(csv_path, 'url')
        labels = read_csv(csv_path, 'label')
        start = time.time()
        for i in range(len(urls)):
            url = urls[i]
            label = labels[i]
            text = get_textofhtml_train(url)
            data = {
                'url':url,
                'text':text,
                'label':label
            }
            df = pd.DataFrame(data, index=range(1))
            df.to_csv('more.csv', sep=',', mode='a', header=False, index=False, encoding='utf-8')#写csv文件，编码为utf-8
            print("Check:%d have been processed!" % (i+1))#记录进度
        end = time.time()
        print("use %d seconds!" % (end - start))#记录时间
