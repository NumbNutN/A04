########################################################
#包：
#time:用以命名
#seleninum：用以打开网站截图
#PIL：打开截图
#pytesseract：分析图片提取文本
#
#注：pytesseract需要先将pytesseract.py文件的tesseract_cmd = 'tesseract.exe'改为
#tesseract_cmd = r'tesseract.exe对应路径'
#################################################################
#函数及其用处：
#timeout_record(url)：将网址写入text文档用于记录
#get_photo(url,path)：获取网站截图
#get_photos_list(url_list,path):以列表形式批处理截图
#get_textofphoto(path):获取先前捕获的截图中文字
###############################################################


import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from PIL import Image
import pytesseract

###############################################################################测试所用变量
url1 = "www.baidu.com"
path = r"./media"
url2 = "github.com"
url3 = "www.bashansy.com"
PathOfChrome = r"./driver/chromedriver_win32/chromedriver.exe"
url_list = [url1,url2]
################################################################################

#全局变量，用于分析
#exa:时间戳暂存变量
#text:截屏所得的文本列表
global exa
global text
##################################
#设置截屏使用的浏览器
#PathOfChrome是驱动绝对路径，此处使用Chrome
####################
driver = webdriver.Chrome(PathOfChrome)
####################

def timeout_record(url):#记录超时的链接，放入特定文件中
    with open('text.txt','a') as f:
        f.write(url)
        f.write("\n")

##################################################################
#                   从单个网站中截屏，保存，访问失败的网站写入text文档
##################################################################

def get_photo(url,path):#url为网址，不必写全http。path为存放截图照片的绝对路径
    frame = "http://" + url + "/"
    try:
        driver.set_page_load_timeout(5)
        driver.get(frame)
        po_path = path + "\\" + "%f.png" % exa
        driver.get_screenshot_as_file(po_path)#采用时间戳命名，可以随截随分析
        driver.close()
    except:
            timeout_record(url)

##################################################################
#                  利用列表批量处理截屏
##################################################################

def get_photos_list(url_list,path):#使用列表的批处理
    for url in url_list:
        get_photo(url,path)

##################################################################
#               输入图片路径，获得图片文字
##################################################################

def get_textofphoto(path):
    po_path = path + "\\" + "%f.png" % exa
    image = Image.open(po_path)
    text = pytesseract.image_to_string(image, lang='chi_sim')
    text = text.replace("“","").replace("。","").replace(" ","").replace("\n","")


# TEST LGD 2023-3-18
get_photo(url1,path)
get_textofphoto(path)