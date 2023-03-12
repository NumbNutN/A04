import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

url1 = "www.baidu.com"
path = r"./media"
url2 = "github.com"
PathOfChrome = r"./driver/chromedriver_win32/chromedriver.exe"
url_list = [url1,url2]


driver = webdriver.Chrome(PathOfChrome)

def timeout_record(url):#记录超时的链接，放入特定文件中
    with open('text.txt','a') as f:
        f.write(url)
        f.write("\n")

def get_photo(url,path):#url为网址，不必写全http。path为存放截图照片的绝对路径
    frame = "http://" + url + "/"
    try:
        driver.set_page_load_timeout(5)
        driver.get(frame)
        driver.get_screenshot_as_file(path+ r"/" + "%f.png" % time.time())
        driver.close()
    except:
            timeout_record(url)


def get_phoos_list(url_list,path):#使用列表的批处理
    for url in url_list:
        get_photo(url,path)

get_phoos_list(url_list,path)